"""Repair only source-audio packaging, retaining the exact complete parent video."""
import copy
import json
import os
import shutil
import subprocess

import project
import workflow


def remux(case, spec):
    from production_pipeline import current_issue, require_production
    from regional_composite import stream_hash
    fid = spec['file_id']
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case, state)
        entry = state['files'][fid]
        issue = next((i for i in entry['issues'] if i['id'] == spec['issue_id']), None)
        if not issue or issue['category'] != 'packaging':
            raise ValueError('Source-audio remux requires a bound packaging issue')
        # Reading the original successful result is allowed in maintenance too.
        if issue['status'] in ('pending_recheck','closed'):
            candidate = workflow.verify_candidate(case, fid, workflow.candidate_for(entry, issue['result_candidate_id']))
            if not candidate.get('packaging_repair') or candidate['packaging_repair']['issue_id'] != issue['id']:
                raise ValueError('Issue was consumed by a different repair')
            return dict(candidate, execution_reused=True, new_model_calls=0, new_renders=0, new_remuxes=0)
        require_production(case)
        if not current_issue(case, fid, entry, issue):
            raise ValueError('Packaging issue must bind the exact current parent and inputs')
        if any(j['status'] in ('running','queued') for j in entry['jobs']):
            raise ValueError('Recover original local job before remux')
        parent = workflow.verify_candidate(case, fid, workflow.candidate_for(entry, issue['candidate_id']))
        identity = dict(parent_id=parent['id'], parent_sha256=parent['output_sha256'], issue_id=issue['id'])
        job_id = 'audio-'+workflow.hash_json(identity)[:24]
        if any(j['id'] == job_id for j in entry['jobs']):
            raise ValueError('Inspect the original packaging job and failure; do not remux it again')
        job = dict(id=job_id, file_id=fid, **identity, status='running', owner_pid=os.getpid(),
                   action='remux-source-audio', started_at=workflow.now(), new_model_calls=0, new_cost_cny=0)
        entry['jobs'].append(job)
    try:
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise ValueError('Existing FFmpeg is unavailable')
        item = workflow.item_for(case, fid)
        source, original = project.project_path(item['source']), project.project_path(parent['output'])
        source_video, parent_video = stream_hash(ffmpeg,source), stream_hash(ffmpeg,original)
        timeline = lambda rows: [','.join(row.split(',')[:4]) for row in rows]
        if timeline(source_video) != timeline(parent_video):
            raise ValueError('Audio remux cannot repair missing frames, changed PTS or a cut video tail')
        metadata = project.probe(source)
        if metadata['audio_stream_count'] > 1:
            raise ValueError('Source has multiple audio tracks; this repair cannot silently discard additional tracks')
        has_audio = metadata['audio_present']
        source_audio = stream_hash(ffmpeg,source,True) if has_audio else []
        parent_has_audio = project.probe(original)['audio_present']
        parent_audio = stream_hash(ffmpeg,original,True) if parent_has_audio else []
        if has_audio == parent_has_audio and source_audio == parent_audio:
            raise ValueError('Parent audio is already exact; review the original issue instead of remuxing')
        output = workflow.state_path(case).parent/fid/(job_id+'.mp4')
        output.parent.mkdir(parents=True,exist_ok=True)
        # Copy both streams; no re-encoding, truncation, padding or retiming.
        subprocess.run([ffmpeg,'-v','error','-nostdin','-n','-copyts','-i',str(original),'-i',str(source),
            '-map','0:v:0','-map','1:a:0?','-c','copy','-movflags','+faststart',str(output)],
            capture_output=True,text=True,check=True,timeout=600)
        output_video = stream_hash(ffmpeg,output)
        output_has_audio = project.probe(output)['audio_present']
        output_audio = stream_hash(ffmpeg,output,True) if output_has_audio else []
        if parent_video != output_video or has_audio != output_has_audio or source_audio != output_audio:
            raise ValueError('Remux changed video/PTS or failed exact audio preservation; output retained as failure')
        report = copy.deepcopy(project.read_json(project.project_path(parent['sidecar'])))
        relative = str(output.relative_to(project.ROOT))
        report.update(output=relative,output_sha256=project.digest(output),
            source_frame_count=len(source_video),output_frame_count=len(output_video),timeline_exact=True,
            source_audio_packets=len(source_audio),output_audio_packets=len(output_audio),
            audio_packets_pts_payload_exact=True,audio_presence=dict(source=has_audio,output=output_has_audio),
            packaging_repair=dict(**identity,parent_video_decoded_exact=True,stream_copy=True),
            continuous_playback_reviewed=False,audio_listened=False,
            status='full_length_candidate_pending_visual_review',accepted=False)
        # Parent pixels keep their frozen producer; only this stream-copy step
        # was executed now. Do not relabel old code as newly executed code.
        artifacts = {}
        for path, expected in report.get('artifact_hashes', {}).items():
            archive = workflow.code_archive(path, expected)
            artifacts[str(archive.relative_to(project.ROOT)) if archive else path] = expected
        for name in ('packaging_repair.py','production_pipeline.py','regional_composite.py','project.py','workflow.py'):
            path = project.ROOT/'scripts'/name
            if path.is_file():
                artifacts[str(path.relative_to(project.ROOT))] = project.digest(path)
        artifacts.update({parent['output']:parent['output_sha256'],
            parent['sidecar']:project.digest(project.project_path(parent['sidecar']))})
        report['artifact_hashes'] = artifacts
        project.project_path(relative+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        child = workflow.register_existing(case,fid,job_id,relative,parent['plan'],parent['id'],
            'pending_agent_quality_review','Exact parent video plus copied source audio; final playback/review remains pending.')
        child['packaging_repair'] = identity
        with workflow.locked_state(case) as state:
            entry = state['files'][fid]
            workflow.verify_candidate(case,fid,parent)
            issue = next(i for i in entry['issues'] if i['id'] == spec['issue_id'])
            if not current_issue(case,fid,entry,issue):
                raise ValueError('Parent changed while remuxing; preserve output without activation')
            entry['candidates'].append(child)
            entry['current_candidate_id'] = child['id']
            issue.update(status='pending_recheck',result_candidate_id=child['id'])
            next(j for j in entry['jobs'] if j['id']==job_id).update(status='completed',
                completed_at=workflow.now(),output=relative)
        return dict(child,new_model_calls=0,new_renders=0,new_remuxes=1)
    except Exception as exc:
        with workflow.locked_state(case) as state:
            next(j for j in state['files'][fid]['jobs'] if j['id']==job_id).update(
                status='failed',error=str(exc),completed_at=workflow.now())
        raise
