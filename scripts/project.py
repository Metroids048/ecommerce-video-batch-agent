#!/usr/bin/env python3
"""Offline preparation, source-based regional composition and evidence checks; no generation."""
import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_COMMAND_HASHES = ContextVar('command_hashes', default=None)


@contextmanager
def command_hashes():
    """Reuse hashes only inside one CLI call, never across commands or threads."""
    token = _COMMAND_HASHES.set({})
    try:
        yield
    finally:
        _COMMAND_HASHES.reset(token)


def file_revision(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def digest(path):
    path = Path(path)
    cache = _COMMAND_HASHES.get()
    before = file_revision(path) if cache is not None else None
    key = str(path.resolve()) if cache is not None else None
    if cache is not None and key in cache and cache[key][0] == before:
        return cache[key][1]
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    result = h.hexdigest()
    if cache is not None:
        if file_revision(path) != before:
            raise ValueError('File changed while hashing: ' + path.name)
        cache[key] = (before, result)
    return result


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Path must stay inside this project')
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def probe(path):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('ffmpeg unavailable; no installation was attempted')
    # -i without an output reads headers and exits; no frames or audio are rewritten.
    result = subprocess.run([ffmpeg, '-hide_banner', '-i', str(path)],
                            capture_output=True, text=True, timeout=30)
    text = result.stderr
    match = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', text)
    video_line = next((line for line in text.splitlines() if 'Video:' in line), '')
    dims = re.search(r'\b(\d{2,5})x(\d{2,5})\b', video_line)
    if not match or not dims:
        raise ValueError('Cannot read video duration/dimensions: ' + path.name)
    h, m, s = map(float, match.groups())
    return {'duration_header_seconds': h * 3600 + m * 60 + s,
            'width': int(dims[1]), 'height': int(dims[2]),
            'audio_present': 'Audio:' in text,
            'audio_stream_count': sum('Stream #' in line and 'Audio:' in line for line in text.splitlines()),
            'precision_note': 'ffmpeg header duration rounded; exact frame/PTS and audio packet audit still required'}


def source_inventory(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError('Cannot read source frame inventory: ' + path.name)
        return {'fps': fps, 'frame_count': int(count),
                'scope': 'container/OpenCV inventory, not decoded full-timeline or visual acceptance'}
    finally:
        cap.release()


def intake(case_relative, request_relative):
    """The Agent structures the user's request; users do not fill this internal JSON."""
    target = project_path(case_relative)
    if target.exists():
        raise ValueError('Case already exists; resume it or choose a new batch, never overwrite it')
    request = read_json(project_path(request_relative))
    case = {'schema_version': 1, 'batch_id': request['batch_id'],
            'case_version': request.get('case_version') or datetime.now(timezone.utc).isoformat(),
            'user_input': request['user_input'], 'files': [], 'pilot_segments': [],
            'provenance': request.get('provenance', 'Not supplied; do not infer customer authorization'),
            'status': 'intake_prepared_not_localized', 'accepted': False}
    from production_effort import new_policy
    case['execution_policy'] = new_policy(request.get('workflow_mode', 'production'))
    # Local registration and processing do not grant provider permission. Bind
    # the restriction to this new order, rather than opening global production.
    config_path = ROOT / 'config/models.json'
    config = read_json(config_path) if config_path.is_file() else None
    if config is not None:
        config = {**config, **config.get('scoped_remote_batches', {}).get(case['batch_id'], {})}
    if config is not None and (config.get('execution_mode') == 'maintenance' or config.get('remote_calls_allowed') is not True):
        case['execution_policy']['local_only'] = True
    if 'reuse_only' in request and not isinstance(request['reuse_only'], bool):
        raise ValueError('reuse_only must be a boolean')
    if request.get('reuse_only'):
        if case['execution_policy']['mode'] != 'production':
            raise ValueError('Reuse-only intake consumes finished media, not method research')
        case['execution_policy']['reuse_only'] = True
    for original in request['files']:
        item = dict(original)
        path = project_path(item['source'])
        meta, inventory = probe(path), source_inventory(path)
        item.update(source_sha256=digest(path), duration_seconds=meta['duration_header_seconds'],
                    dimensions=[meta['width'], meta['height']], fps_container_average=inventory['fps'],
                    source_frame_count_inventory=inventory['frame_count'],
                    audio_detected=meta['audio_present'], audio_required=item.get('audio_required', meta['audio_present']),
                    title=item.get('title', item['id']),
                    operation_ids=[op['id'] for op in item['operations']])
        # Localization and quality claims are produced by later execution, never inherited from a request.
        item['operations'] = [dict(op, occurrences=None, full_coverage_verified=False) for op in item['operations']]
        case['files'].append(item)
    validate_case(case)
    target.parent.mkdir(parents=True, exist_ok=True)
    from source_timeline import build
    for item in case['files']:
        timeline_path = target.parent / (item['id'] + '-timeline.json')
        if timeline_path.exists():
            raise ValueError('Timeline exists; preserve the prior intake attempt')
        timeline_path.write_text(json.dumps(build(project_path(item['source']))))
        item.update(source_timeline=str(timeline_path.relative_to(ROOT)),
                    source_timeline_sha256=digest(timeline_path))
    with target.open('x', encoding='utf-8') as stream:
        json.dump(case, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return {'status': case['status'], 'case': str(target.relative_to(ROOT)),
            'input_count': len(case['files']), 'new_model_calls': 0, 'accepted': False}


def validate_case(case, verify_hash=True):
    files = case.get('files', [])
    if not case.get('batch_id') or not files:
        raise ValueError('Missing batch_id or input files')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', case['batch_id']):
        raise ValueError('Invalid batch_id; use a unique directory-safe identifier')
    if not case.get('case_version'):
        raise ValueError('Missing case_version')
    if not project_path(case['user_input']).is_file():
        raise ValueError('Missing actual natural-language input file')
    ids, metadata = set(), {}
    for item in files:
        file_id = item['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', file_id):
            raise ValueError('Invalid file id')
        if file_id in ids:
            raise ValueError('Duplicate file id: ' + file_id)
        ids.add(file_id)
        path = project_path(item['source'])
        if not path.is_file():
            raise ValueError('Missing input: ' + file_id)
        if verify_hash and digest(path) != item['source_sha256']:
            raise ValueError('Source hash changed: ' + file_id)
        meta = probe(path)
        if item.get('audio_required') and not meta['audio_present']:
            raise ValueError('Original audio required but absent: ' + file_id)
        operations = item.get('operations', [])
        if not operations:
            raise ValueError('No operation specified: ' + file_id)
        op_ids = set()
        for op in operations:
            if not op.get('target') or not op.get('protect'):
                raise ValueError('Target/protection requirements missing: ' + file_id)
            if op['id'] in op_ids:
                raise ValueError('Duplicate operation: ' + file_id)
            op_ids.add(op['id'])
            if op.get('kind') not in ('remove', 'replace', 'modify', 'repair'):
                raise ValueError('Unsupported operation kind')
            if op['kind'] == 'modify' and not op.get('desired_change'):
                raise ValueError('Element modification needs an explicit desired_change')
            if op.get('reference_asset') and not project_path(op['reference_asset']).is_file():
                raise ValueError('Reference asset missing: ' + file_id)
            if op['kind'] == 'replace':
                asset = op.get('replacement_asset')
                if not asset or not project_path(asset).is_file():
                    raise ValueError('Replacement asset missing: ' + file_id)
        metadata[file_id] = meta
    for segment in case.get('pilot_segments', []):
        item = next((f for f in files if f['id'] == segment['file_id']), None)
        if item is None or segment['operation_id'] not in item['operation_ids']:
            raise ValueError('Pilot references wrong file/operation')
        start, end = segment['source_start_seconds'], segment['source_end_seconds']
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)):
            raise ValueError('Pilot interval must be finite numeric seconds')
        if not 0 <= start < end <= item['duration_seconds']:
            raise ValueError('Pilot interval outside complete source: ' + segment['id'])
    return metadata


def prepare(case):
    metadata = validate_case(case)
    model = read_json(ROOT / 'config/models.json')['primary']
    tasks = []
    for segment in case.get('pilot_segments', []):
        item = next(f for f in case['files'] if f['id'] == segment['file_id'])
        op = next(o for o in item['operations'] if o['id'] == segment['operation_id'])
        length = segment['source_end_seconds'] - segment['source_start_seconds']
        limits = model['limits']
        if length > limits['max_reference_video_seconds_conservative'] or 2 * length > limits['max_input_plus_output_seconds']:
            raise ValueError('Pilot exceeds selected model limits')
        parameters = dict(model['pilot_parameters'], duration=str(length))
        key_data = {'source_sha256': item['source_sha256'], 'segment': segment,
                    'operation': op, 'model_id': model['id'], 'parameters': parameters}
        key = hashlib.sha256(json.dumps(key_data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        tasks.append({'pilot_id': segment['id'], 'file_id': item['id'],
                      'source_sha256': item['source_sha256'],
                      'source_time_seconds': [segment['source_start_seconds'], segment['source_end_seconds']],
                      'operation_id': op['id'], 'target': op['target'], 'protect': op['protect'],
                      'route': model['route'], 'parameter_draft': parameters,
                      'cache_key': key, 'provider_task_id': None,
                      'status': 'draft_not_submitted',
                      'blocks': ['input interval visual confirmation', 'actual clip extraction and hash',
                                 'current legal contract/quote', 'controlled upload and task-reuse check']})
    return {'batch_id': case['batch_id'], 'case_version': case['case_version'],
            'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_no_generation', 'ready_for_cloud_submit': False,
            'accepted': False, 'generation_calls': 0, 'actual_generation_cny': 0,
            'input_count': len(case['files']), 'input_probes': metadata,
            'full_target_localization_verified': False,
            'tasks': tasks, 'note': 'Offline preflight only. No API, edits, playback, or quality acceptance.'}


def repair_plan(case, file_id, start, end, parent, category='protection'):
    from workflow import resolve_parent, dependency_plan
    item = next((f for f in case['files'] if f['id'] == file_id), None)
    if item is None:
        raise ValueError('Unknown source file id')
    if not parent or not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end <= item['duration_seconds']:
        raise ValueError('Repair must bind a parent and a valid source interval')
    candidate = resolve_parent(case, file_id, parent)
    result = dependency_plan(case, candidate, [start, end], category)
    result.update(batch_id=case['batch_id'],
                  source_context_seconds=[max(0, start-.5), min(item['duration_seconds'], end+.5)],
                  reuse_other_file_ids=[f['id'] for f in case['files'] if f['id'] != file_id],
                  reuse_same_file_outside_seconds=[[0,start],[end,item['duration_seconds']]])
    return result


def check_delivery(case):
    validate_case(case)
    from workflow import state_path, delivery_report
    if state_path(case).exists():
        report = delivery_report(case)
        issues = []
        for entry in report['files']:
            if entry['delivery_status'] != 'qualified' or entry['unresolved_issues']:
                issues.append(entry['id'] + ': ' + entry['delivery_status'] + '; unresolved review/requirements')
        return dict(report, packaging_checks_passed=not issues, issues=issues,
                    note='Canonical workflow binding/status check. No automatic visual judgment or user acceptance.')
    path = ROOT / 'deliveries' / case['batch_id'] / 'manifest.json'
    issues = []
    if not path.is_file():
        issues.append('No delivery manifest or new complete edited videos')
        return {'batch_id': case['batch_id'], 'packaging_checks_passed': False,
                'accepted': False, 'issues': issues}
    manifest = read_json(path)
    if manifest.get('case_version') != case['case_version']:
        issues.append('Delivery belongs to another case version')
    entries = manifest.get('files', [])
    if len({e.get('id') for e in entries}) != len(entries):
        issues.append('Duplicate delivery file ids')
    if {e.get('id') for e in entries} != {f['id'] for f in case['files']}:
        issues.append('Delivery input/output mapping incomplete or extra files')
    for source in case['files']:
        entry = next((e for e in entries if e.get('id') == source['id']), None)
        if entry is None:
            continue
        if entry.get('source_sha256') != source['source_sha256']:
            issues.append(source['id'] + ': source hash/version mismatch')
        output = project_path(entry.get('output', 'missing'))
        if not output.is_file():
            issues.append(source['id'] + ': missing complete output')
            continue
        output_hash = digest(output)
        if entry.get('output_sha256') != output_hash:
            issues.append(source['id'] + ': output hash mismatch')
        meta = probe(output)
        if abs(meta['duration_header_seconds'] - source['duration_seconds']) > 1 / source['fps_container_average'] + 0.01:
            issues.append(source['id'] + ': duration changed; exact PTS audit required')
        if source['audio_required'] and not meta['audio_present']:
            issues.append(source['id'] + ': original audio missing')
        for gate in ['target_coverage', 'full_visual_review', 'timeline', 'audio_preservation']:
            relative = entry.get('evidence', {}).get(gate)
            if not relative or not project_path(relative).is_file():
                issues.append(source['id'] + ': missing ' + gate + ' evidence')
                continue
            evidence = read_json(project_path(relative))
            if (evidence.get('source_sha256') != source['source_sha256']
                    or evidence.get('output_sha256') != output_hash
                    or evidence.get('passed') is not True):
                issues.append(source['id'] + ': stale/failed ' + gate + ' evidence')
    return {'batch_id': case['batch_id'], 'packaging_checks_passed': not issues,
            'accepted': False, 'issues': issues,
            'note': 'Evidence presence/binding and basic metadata checks only; visual/audio evidence must come from actual review. User acceptance remains separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['pipeline', 'intake', 'prepare', 'matte', 'stage', 'preview-review', 'check', 'repair-plan', 'compose', 'execute', 'preview', 'localize', 'refine', 'feedback', 'agent-review', 'review-init', 'review', 'serve', 'observe', 'deliver', 'audit-text', 'provider', 'runtime'])
    parser.add_argument('--case', required=True)
    parser.add_argument('--out')
    parser.add_argument('--diagnostic', action='store_true', help='Export explicit failure evidence, never a qualified delivery')
    parser.add_argument('--request', help='Agent-written internal intake JSON; preserve the original user_input separately')
    parser.add_argument('--spec', help='Source-bound visual observations supplied by the host Agent')
    parser.add_argument('--file')
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    parser.add_argument('--parent')
    parser.add_argument('--plan')
    parser.add_argument('--candidate')
    parser.add_argument('--issue')
    parser.add_argument('--category', default='protection', choices=['localization','asset','protection','composition','packaging','model_internal'])
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    exit_code = 2
    try:
        if args.command in ('serve', 'review', 'runtime'):
            exit_code = dispatch_command(args)
        else:
            with command_hashes():
                exit_code = dispatch_command(args)
        return exit_code
    finally:
        # An append-only measurement, never a second production state. No spec,
        # prompt, credentials or provider URLs are copied into this journal.
        if args.command not in ('serve', 'review', 'runtime', 'check'):
            try:
                case = read_json(project_path(args.case))
                batch = case['batch_id']
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', batch):
                    raise ValueError('Invalid batch id for timing')
                journal = ROOT / 'runs' / batch / 'timings.jsonl'
                journal.parent.mkdir(parents=True, exist_ok=True)
                with journal.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(command=args.command,
                        started_at=started_at, completed_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=round(time.monotonic()-started, 6), exit_code=exit_code,
                        candidate_id=args.candidate, **timing_context(args, case)), ensure_ascii=False)+'\n')
            except (ValueError, KeyError, OSError) as exc:
                print('Stage timing unavailable: '+str(exc), file=sys.stderr)


def record_stage_timing(case, spec):
    """Measure actual host intervals; method starts also arm the bound effort clock."""
    import fcntl
    import uuid
    batch = case['batch_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', batch):
        raise ValueError('Invalid batch id for timing')
    journal = ROOT / 'runs' / batch / 'stage-events.jsonl'
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        events = [json.loads(line) for line in stream if line.strip()]
        at = datetime.now(timezone.utc).isoformat()
        if spec['action'] == 'start':
            if spec['phase'] not in ('source_observation', 'localization', 'matte', 'preview_review', 'final_review',
                                      'engineering', 'source_selection', 'production_session', 'provider_wait', 'packaging'):
                raise ValueError('Unknown measured stage')
            item = next((f for f in case['files'] if f['id'] == spec.get('file_id')), None)
            if spec.get('file_id') and item is None:
                raise ValueError('Stage file does not belong to case')
            if spec.get('operation_id') and (item is None or spec['operation_id'] not in [o['id'] for o in item['operations']]):
                raise ValueError('Stage operation does not belong to file')
            if spec['phase'] in ('localization', 'matte'):
                if case.get('execution_policy') and item is None:
                    raise ValueError('Method work needs a file id for its feasibility clock')
                from production_effort import check as check_effort
                check_effort(case, item['id'] if item else None, start_method=bool(item))
            event = dict(event='phase_start', event_id=uuid.uuid4().hex, at=at,
                batch_id=batch, case_version=case['case_version'], phase=spec['phase'],
                work_class={'engineering':'one_time_engineering', 'source_selection':'research_selection'}.get(spec['phase'], 'per_video'),
                file_id=item['id'] if item else None, operation_id=spec.get('operation_id'),
                source_sha256=item['source_sha256'] if item else None)
        elif spec['action'] == 'end':
            starts = [e for e in events if e.get('event') == 'phase_start' and e.get('event_id') == spec['event_id']]
            if (len(starts) != 1 or starts[0]['case_version'] != case['case_version'] or
                    any(e.get('event') == 'phase_end' and e.get('event_id') == spec['event_id'] for e in events)):
                raise ValueError('Stage start missing, changed or already ended')
            start = starts[0]
            elapsed = (datetime.fromisoformat(at)-datetime.fromisoformat(start['at'])).total_seconds()
            if elapsed < 0:
                raise ValueError('Stage clock moved backwards')
            event = dict(start, event='phase_end', at=at, elapsed_seconds=elapsed)
        else:
            raise ValueError('Stage action must be start or end')
        stream.seek(0, 2)
        stream.write(json.dumps(event, ensure_ascii=False)+'\n')
    return event


def timing_summary(case, until=None):
    """Report measured intervals, overlap and gaps; never reconstruct unrecorded labor."""
    directory=ROOT/'runs'/case['batch_id']
    read_rows=lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.is_file() else []
    commands=read_rows(directory/'timings.jsonl');events=read_rows(directory/'stage-events.jsonl')
    end=datetime.fromisoformat(until) if until else datetime.now(timezone.utc)
    intervals=[];sessions=[];bounds=[];open_stages=[]
    def add(start, stop, work_class, phase):
        a=datetime.fromisoformat(start);b=min(datetime.fromisoformat(stop),end)
        if b<a:
            raise ValueError('Timing interval moved backwards')
        bounds.extend((a,b))
        (sessions if phase=='production_session' else intervals).append((a,b,work_class,phase))
    for command in commands:
        add(command['started_at'],command['completed_at'],command.get('work_class','unclassified'),command['command'])
    ends={e['event_id']:e for e in events if e.get('event')=='phase_end'}
    for start in (e for e in events if e.get('event')=='phase_start'):
        finish=ends.get(start['event_id'])
        if not finish:
            open_stages.append(dict(event_id=start['event_id'],phase=start['phase']))
        add(start['at'],finish['at'] if finish else end.isoformat(),start.get('work_class','unclassified'),start['phase'])
    def union(rows):
        merged=[]
        for a,b,*_ in sorted(rows):
            if merged and a<=merged[-1][1]:
                merged[-1][1]=max(merged[-1][1],b)
            else:
                merged.append([a,b])
        return round(sum((b-a).total_seconds() for a,b in merged),6)
    earliest=min(bounds) if bounds else None
    wall=round((end-earliest).total_seconds(),6) if earliest else None
    active=union(intervals)
    return dict(scope='Earliest recorded event to summary; earlier unrecorded work is unknown. Production-session envelopes are not activity attribution.',
        started_at=earliest.isoformat() if earliest else None,completed_at=end.isoformat(),
        observed_wall_seconds=wall,recorded_activity_union_seconds=active,
        unattributed_wall_seconds=round(max(0,wall-active),6) if wall is not None else None,
        work_class_union_seconds={k:union([r for r in intervals if r[2]==k]) for k in sorted({r[2] for r in intervals})},
        phase_union_seconds={k:union([r for r in intervals if r[3]==k]) for k in sorted({r[3] for r in intervals})},
        production_session_union_seconds=union(sessions),open_stages=open_stages,
        command_count=len(commands),failed_commands=sum(c.get('exit_code',0)!=0 for c in commands),
        overlap_note='Phase and work-class subtotals can overlap; do not add them. Failed commands and repairs remain included.')


def timing_context(args, case):
    """Attribute commands without logging prompts, provider URLs or full specs."""
    body = {}
    for name in ('spec', 'request', 'plan'):
        value = getattr(args, name, None)
        if value:
            try:
                body = read_json(project_path(value))
            except (ValueError, OSError):
                pass
            break
    if not isinstance(body, dict):
        body = {}
    fid = args.file or body.get('file_id')
    item = next((f for f in case['files'] if f['id'] == fid), None)
    config_path = ROOT / 'config/models.json'
    mode = read_json(config_path).get('execution_mode') if config_path.is_file() else None
    result = dict(file_id=fid if item else None,
                  work_class='one_time_engineering' if mode == 'maintenance' else 'per_video')
    if item and body.get('operation_id') in [o['id'] for o in item['operations']]:
        result['operation_id'] = body['operation_id']
    if args.command == 'provider':
        if body.get('action') in ('prepare', 'replay', 'review-raw', 'plan', 'submit', 'query', 'download',
                                 'prepare-recolor', 'prepare-native-recolor', 'plan-native-recolor', 'prepare-native-edit', 'plan-native-edit',
                                 'import-segmentation', 'plan-source-recolor', 'collect-native-recolor'):
            result['provider_action'] = body['action']
        if re.fullmatch(r'[A-Za-z0-9_-]{1,96}', str(body.get('unit_id', ''))):
            result['unit_id'] = body['unit_id']
    return result


def dispatch_command(args):
    try:
        from production_pipeline import require_production, guard_upstream
        diagnostic_local = args.diagnostic and args.command in ('execute', 'deliver')
        if args.diagnostic and args.command == 'provider' and args.spec:
            provider_spec = read_json(project_path(args.spec))
            diagnostic_local = provider_spec.get('action') in ('review-raw', 'plan-native-recolor') or (
                provider_spec.get('action') == 'plan' and provider_spec.get('diagnostic') is True)
        if args.command == 'intake':
            if not args.request:
                raise ValueError('intake requires --request')
            # Intake binds local-only work; rendering still checks method and quality.
            print(json.dumps(intake(args.case, args.request), ensure_ascii=False))
            return 0
        case = read_json(project_path(args.case))
        reuse_only = case.get('execution_policy', {}).get('reuse_only') is True
        if reuse_only and args.command in ('prepare', 'matte', 'preview', 'preview-review', 'execute',
                'compose', 'localize', 'refine', 'provider'):
            require_production(case)
        if not diagnostic_local and args.command in ('prepare', 'matte', 'preview', 'execute', 'compose',
                'localize', 'refine', 'observe', 'deliver', 'provider', 'review-init'):
            if not (reuse_only and args.command in ('review-init', 'deliver', 'observe')):
                require_production(case)
        if args.command == 'pipeline':
            from production_pipeline import dispatch
            spec = read_json(project_path(args.spec)) if args.spec else {'action': 'status'}
            print(json.dumps(dispatch(case, spec), ensure_ascii=False))
            return 0
        if args.command in ('matte', 'localize', 'refine'):
            spec = read_json(project_path(args.spec)) if args.spec else {}
            guard_upstream(case, args.file or spec.get('file_id'), operation_id=spec.get('operation_id'))
        if args.command == 'preview-review':
            if not args.spec:
                raise ValueError('preview-review requires --spec with actual observations')
            from production_effort import review_preview
            print(json.dumps(review_preview(case, read_json(project_path(args.spec))), ensure_ascii=False))
            return 0
        if args.command in ('localize', 'refine'):
            from production_effort import check as check_effort
            check_effort(case, args.file, start_method=bool(args.file))
        if args.command == 'stage':
            if not args.spec:
                raise ValueError('stage requires --spec with start/end action')
            print(json.dumps(record_stage_timing(case, read_json(project_path(args.spec))), ensure_ascii=False))
            return 0
        if args.command == 'matte':
            if not args.spec or not args.out:
                raise ValueError('matte requires source-bound --spec and new --out .npz')
            from source_matte import prepare
            result = prepare(case, read_json(project_path(args.spec)), args.out)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        elif args.command == 'serve':
            from review_server import serve
            return serve(case, args.case, args.port)
        elif args.command == 'provider':
            if not args.spec:raise ValueError('provider requires Agent-prepared --spec')
            from runninghub_adapter import dispatch
            result = dispatch(case,read_json(project_path(args.spec)))
        elif args.command == 'runtime':
            from runtime_checkpoint import capture
            result = capture(case)
        elif args.command == 'review-init':
            from workflow import initialize
            result = initialize(case, args.case)
        elif args.command == 'review':
            from workflow import delivery_report
            result = delivery_report(case)
        elif args.command == 'deliver':
            if not args.out:
                raise ValueError('deliver requires a new directory via --out')
            from delivery_package import build
            options=read_json(project_path(args.spec)) if args.spec else {}
            result = build(case, args.out, diagnostic=args.diagnostic,
                           full_comparison=options.get('full_comparison',False))
            print(json.dumps(result, ensure_ascii=False))
            return 0
        elif args.command == 'observe':
            if not args.file or not args.out:
                raise ValueError('observe requires --file --out')
            observation = read_json(project_path(args.spec)) if args.spec else {}
            if observation.get('mode')=='all-frame-pairs':
                from review_queue import build_pairs
                result=build_pairs(case,args.file,args.out,observation,candidate_id=args.candidate)
            elif observation.get('mode')=='native-playback':
                from review_queue import native_playback
                result=native_playback(case,args.file,args.out,candidate_id=args.candidate)
            else:
                from review_queue import build
                result = build(case, args.file, args.out, plan=read_json(project_path(args.plan)) if args.plan else None,
                               frames=observation.get('frames'), stride=observation.get('stride', 30), candidate_id=args.candidate)
            # observe writes its images and queue inside --out, unlike JSON report commands.
            print(json.dumps(result, ensure_ascii=False))
            return 0
        elif args.command == 'prepare':
            result = prepare(case)
        elif args.command == 'check':
            result = check_delivery(case)
        elif args.command == 'audit-text':
            if not args.spec or not args.out:
                raise ValueError('audit-text requires --spec --out')
            from text_localization import audit_output
            result = audit_output(case, read_json(project_path(args.spec)), args.out)
            print(json.dumps(result, ensure_ascii=False))
            return 2 if result['status'] == 'fail' else 0
        elif args.command in ('localize', 'refine'):
            if not args.spec or not args.out or (args.command == 'refine' and not args.plan):
                raise ValueError('localize/refine require --spec --out; refine also requires --plan')
            from planar_text import localize, refine
            spec = read_json(project_path(args.spec))
            result = (localize(case, spec, args.out) if args.command == 'localize' else
                      refine(case, read_json(project_path(args.plan)), spec, args.out))
            print(json.dumps(result, ensure_ascii=False))
            return 0
        elif args.command == 'feedback':
            if not args.request:
                raise ValueError('feedback requires --request')
            from workflow import add_feedback
            result = add_feedback(case, read_json(project_path(args.request)))
        elif args.command == 'agent-review':
            if not args.request:
                raise ValueError('agent-review requires a bound observation JSON via --request')
            from workflow import record_agent_review
            result = record_agent_review(case, args.request)
        elif args.command == 'preview':
            if not args.plan or not args.out or None in (args.start,args.end):
                raise ValueError('preview requires --plan --out --start --end (source seconds)')
            from execution_plan import preview
            result=preview(case,read_json(project_path(args.plan)),project_path(args.out),args.start,args.end)
            print(json.dumps(result,ensure_ascii=False))
            return 0
        elif args.command == 'execute':
            if not args.plan or not args.candidate:
                raise ValueError('execute requires --plan and --candidate; repairs also require --parent and --issue')
            from workflow import execute_plan
            result = execute_plan(case, args.plan, args.candidate, args.parent, args.issue, diagnostic=args.diagnostic)
        elif args.command == 'compose':
            if not args.plan or not args.out:
                raise ValueError('compose requires --plan and --out (video path)')
            from regional_composite import render
            result = render(case, read_json(project_path(args.plan)), project_path(args.out))
            project_path(args.out + '.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({k: result[k] for k in ('output', 'source_frame_count', 'output_frame_count', 'timeline_exact', 'audio_packets_pts_payload_exact', 'status', 'accepted')}, ensure_ascii=False))
            return 2 if result['status'] == 'failed_integrity_check' else 0
        else:
            validate_case(case)
            if None in (args.start, args.end) or not args.file or not args.parent:
                raise ValueError('repair-plan requires --file --start --end --parent')
            result = repair_plan(case, args.file, args.start, args.end, args.parent, args.category)
        output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.out:
            target = project_path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(output, encoding='utf-8')
            print(json.dumps({'written': str(target.relative_to(ROOT)),
                              'status': result.get('status'), 'accepted': False}, ensure_ascii=False))
        else:
            print(output, end='')
        return 2 if args.command == 'check' and not result['packaging_checks_passed'] else 0
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({'status': 'blocked', 'error': str(exc), 'accepted': False}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    sys.exit(main())
