"""Complete per-file operation manifests; local assets and source-mapped provider pixels."""
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
import project


def coverage_records(plan, count):
    records=plan['review']['records']
    states=plan['review']['frame_states']
    if (len(states)!=count or [r.get('frame') for r in records]!=list(range(count))
            or [r.get('state') for r in records]!=states):
        raise ValueError('Complete ordered coverage records must match every source frame/state')
    return records


def coverage_findings(records):
    """Expose unedited detections and short holes between the same source plane.

    This is a conservative dropout warning, not proof of target absence or
    visual quality. Unknown frames remain unresolved in the operation report.
    """
    import workflow
    known=[]; written=[]
    for record in records:
        for instance in record.get('instances',[record]):
            quad=instance.get('word_quad',instance.get('source_quad'))
            if quad is None:continue
            if instance.get('state',record.get('state'))=='visible':
                written.append((record['frame'],np.float32(quad)))
            else:
                known.append(dict(frame=record['frame'],instance_id=instance.get('instance_id'),
                                  reason=instance.get('blocked_reason','unresolved target')))
    holes=set()
    def same_plane(q, other):
        overlap,_=cv2.intersectConvexConvex(cv2.convexHull(q),cv2.convexHull(other))
        length=max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[1]))
        return (overlap/max(1,min(abs(cv2.contourArea(q)),abs(cv2.contourArea(other))))>.2
                and np.linalg.norm(q.mean(axis=0)-other.mean(axis=0))<length*.3)
    for index,(n,q) in enumerate(written):
        for m,p in written[index+1:]:
            if m-n>4:break
            if m-n<2:continue
            overlap,_=cv2.intersectConvexConvex(cv2.convexHull(q),cv2.convexHull(p))
            if overlap/max(1,min(abs(cv2.contourArea(q)),abs(cv2.contourArea(p))))<.5:continue
            for missing in range(n+1,m):
                if not any(k==missing and (same_plane(q,other) or same_plane(p,other))
                           for k,other in written if k==missing):
                    holes.add(missing)
    return {'known_unedited_instances':known,'replacement_dropout_risks':workflow.ranges(sorted(holes))}


def require_frozen_appearances(case, manifest):
    """Validate A's full scope in every compositor path, including diagnostics."""
    import workflow
    path = workflow.state_path(case)
    if not path.is_file():
        return
    state = project.read_json(path)
    workflow.verify_state_case(case, state)
    file_entry = state['files'][manifest['file_id']]
    route = file_entry.get('route_handoff')
    if not route:
        if case.get('execution_policy', {}).get('stage_handoff_required'):
            raise ValueError('Freeze source observation and route before preparation, preview or submit')
        return
    item = workflow.item_for(case, manifest['file_id'])
    if route['dependency'] != workflow.requirement_binding(case, item):
        raise ValueError('Frozen route inputs changed')
    if project.digest(project.project_path(route['observation']['path'])) != route['observation']['sha256']:
        raise ValueError('Frozen source observation changed')
    observed = project.read_json(project.project_path(route['observation']['path']))
    from source_observation import require_receipt, require_plan_instances
    receipt = require_receipt(case, manifest['file_id'], file_entry, route['observation'])
    for entry in manifest['operations']:
        if entry.get('disposition') != 'ready':
            continue  # Blocked work stays explicit and cannot qualify at delivery.
        plan_path = project.project_path(entry['plan'])
        if project.digest(plan_path) != entry['plan_sha256']:
            raise ValueError('Operation plan changed')
        plan = project.read_json(plan_path)
        states = plan['review']['frame_states']
        missing = [n for lo,hi in observed['operations'][entry['operation_id']]['appearance_ranges_half_open']
                   for n in range(lo,hi) if n >= len(states) or states[n] not in ('visible','partial','occluded','absent')]
        if missing:
            raise ValueError('Frozen appearances missing from operation ' + entry['operation_id'] +
                             '; source frame ranges ' + str(workflow.ranges(missing)))
        # Legacy ranges retain their original visible/occluded rule. New
        # receipts distinguish visible targets from fully occluded and absent.
        if receipt is None and any(states[n] not in ('visible','occluded')
                for lo,hi in observed['operations'][entry['operation_id']]['appearance_ranges_half_open'] for n in range(lo,hi)):
            raise ValueError('Frozen appearances missing from operation ' + entry['operation_id'])
        require_plan_instances(receipt, entry['operation_id'], plan)


def validate(case, manifest):
    import workflow
    item = workflow.item_for(case, manifest['file_id'])
    if manifest.get('schema') != 'execution_manifest_v1' or manifest.get('dependency') != workflow.requirement_binding(case, item):
        raise ValueError('Manifest source/requirements/assets binding mismatch')
    requested = {op['id']: op for op in item['operations']}
    entries = manifest['operations']
    if len(entries) != len(requested) or {op['operation_id'] for op in entries} != set(requested):
        raise ValueError('Manifest must include every requested operation exactly once')
    require_frozen_appearances(case, manifest)
    from source_timeline import load_for_item, require_cfr
    timeline = load_for_item(item)
    require_cfr(timeline)
    artifacts = {}
    loaded = []
    for entry in entries:
        op = requested[entry['operation_id']]
        if entry['disposition'] == 'blocked':
            if not entry.get('blocked_reason'):
                raise ValueError('Blocked operation needs reason')
            loaded.append((entry, None)); continue
        if entry['disposition'] != 'ready':
            raise ValueError('Unsupported operation disposition')
        path = project.project_path(entry['plan'])
        if project.digest(path) != entry['plan_sha256']:
            raise ValueError('Operation plan changed')
        plan = project.read_json(path)
        if (plan['file_id'] != item['id'] or plan['source_sha256'] != item['source_sha256'] or
                plan['case_version'] != case['case_version'] or plan['operation_id'] != op['id']):
            raise ValueError('Operation plan belongs to another request')
        from production_pipeline import guard_method
        guard_method(case, item['id'], op['id'], plan=plan)
        from production_effort import source_preserving_recolor, provider_first
        from production_pipeline import interval_route
        interval = interval_route(case,item['id'],op['id'])
        if interval and plan.get('composition_mode') != 'aigc_regional':
            raise ValueError('Fixed interval edit must consume the reviewed model pixels')
        if source_preserving_recolor(op) and not provider_first(case) and not interval and plan.get('composition_mode') != 'source_recolor':
            raise ValueError('Frozen source-preserving route cannot switch to another compositor')
        if plan.get('composition_mode') == 'cached_clean_plate':
            from cached_clean_plate import validate as validate_plate
            artifacts.update(validate_plate(case,item,plan))
            artifacts[entry['plan']] = entry['plan_sha256']
            if plan['source_frame_count'] != len(timeline['pts']):
                raise ValueError('Clean plate plan needs complete source inventory')
            coverage_records(plan,len(timeline['pts']))
            loaded.append((entry,plan));continue
        if plan.get('composition_mode') == 'source_recolor':
            from source_recolor import validate as validate_recolor
            artifacts.update(validate_recolor(case,item,plan))
            artifacts[entry['plan']] = entry['plan_sha256']
            if plan['source_frame_count'] != len(timeline['pts']) or plan['source_frames_half_open'] != [0,len(timeline['pts'])]:
                raise ValueError('Recolor plan must inventory the complete source')
            coverage_records(plan,len(timeline['pts']))
            loaded.append((entry,plan));continue
        if plan.get('composition_mode') == 'aigc_regional':
            from aigc_composite import validate as validate_aigc
            artifacts.update(validate_aigc(case,item,plan))
            artifacts[entry['plan']] = entry['plan_sha256']
            if plan['source_frame_count'] != len(timeline['pts']) or plan['source_frames_half_open'] != [0,len(timeline['pts'])]:
                raise ValueError('AIGC plan must cover the complete source inventory')
            coverage_records(plan,len(timeline['pts']))
            loaded.append((entry,plan));continue
        if (plan.get('composition_mode') != 'exact_asset_on_source_label' or plan.get('mask_space') != 'label_plane'):
            raise ValueError('Unified executor currently supports exact assets on source planes only')
        if plan['exact_asset'] != op.get('replacement_asset'):
            raise ValueError('Asset does not belong to this exact operation')
        for key in ('masks', 'exact_asset') + tuple(k for k in ('localization_evidence','carrier_evidence','composition_implementation','geometry_parent') if plan.get(k)):
            if project.digest(project.project_path(plan[key])) != plan[key+'_sha256']:
                raise ValueError('Operation artifact changed: '+key)
            artifacts[plan[key]] = plan[key+'_sha256']
        artifacts[entry['plan']] = entry['plan_sha256']
        count = len(timeline['pts'])
        if plan['source_frame_count'] != count or plan['source_frames_half_open'] != [0, count]:
            raise ValueError('Full-video request needs a full source timeline plan')
        coverage_records(plan, count)
        loaded.append((entry, plan))
    return item, timeline, loaded, artifacts


def load_layers(loaded,timeline):
    import workflow
    layers = []
    reports = []
    for entry, plan in loaded:
        report = dict(operation_id=entry['operation_id'], disposition=entry['disposition'],
                      blocked_reason=entry.get('blocked_reason'), modified_source_frames=[],
                      modified_instance_count=0, unresolved_ranges=[])
        reports.append(report)
        if plan is None:
            report['unresolved_ranges'] = [[0,len(timeline['pts'])]]; continue
        if plan.get('composition_mode') == 'cached_clean_plate':
            from cached_clean_plate import Layer
            layers.append(Layer(plan,report));continue
        if plan.get('composition_mode') == 'source_recolor':
            from source_recolor import Layer
            layers.append(Layer(plan,report));continue
        if plan.get('composition_mode') == 'aigc_regional':
            from aigc_composite import Layer
            report['unresolved_ranges'] = workflow.ranges([i for i,s in enumerate(plan['review']['frame_states']) if s=='unresolved'])
            layers.append(Layer(plan,report));continue
        with np.load(project.project_path(plan['masks']), allow_pickle=False) as z:
            masks = {k:z[k] for k in z.files}
        records = {r['frame']:r for r in plan['review']['records']}
        report.update(coverage_findings(plan['review']['records']))
        report['unresolved_ranges'] = workflow.ranges([i for i,s in enumerate(plan['review']['frame_states']) if s in ('unresolved', 'partial')])
        asset = cv2.imread(str(project.project_path(plan['exact_asset'])), cv2.IMREAD_UNCHANGED)
        if asset is None:
            raise ValueError('Cannot decode exact asset')
        layers.append((plan,masks,records,asset,report))
    return layers,reports


def compose_layers(frame,n,layers):
    from surface_text import compose_planar_text
    from text_localization import require_visible_word
    from regional_composite import compose_frame
    height,width=frame.shape[:2]
    result=frame.copy(); union=np.zeros((height,width),bool)
    for layer in layers:
        if hasattr(layer,'compose'):
            patch,touched=layer.compose(frame,n)
            if layer.plan['review']['frame_states'][n] == 'visible' and not np.any(touched):
                raise ValueError(f'Visible occurrence has no effective write at source frame {n}')
            if np.any(union & touched):raise ValueError(f'Unresolved operation overlap at source frame {n}')
            result[touched]=patch[touched];union|=touched
            continue
        plan,masks,records,asset,report=layer
        record=records.get(n,{})
        if record.get('state') not in ('visible','partial'):
            continue
        for instance in record.get('instances', [record]):
            if instance.get('state', 'visible') not in ('visible','partial'):
                continue
            if 'source_quad' not in instance or not np.any(masks['alpha'][n]):
                raise ValueError('Visible occurrence has no usable track/mask')
            pw,ph=plan['label_style']['plane_size']
            plane=np.float32([[0,0],[pw-1,0],[pw-1,ph-1],[0,ph-1]])
            matrix=cv2.getPerspectiveTransform(plane,np.float32(instance['source_quad']))
            alpha=cv2.warpPerspective(masks['alpha'][n],matrix,(width,height))
            protection=cv2.warpPerspective(masks['protected'][n],matrix,(width,height)) if 'protected' in masks else None
            style=dict(plan['label_style'],**record.get('label_style',{}))
            style.update(instance.get('label_style',{}))
            if 'word_quad' in instance:
                partial=(style.get('projected_partial',False) and
                         instance.get('track_method') in ('source_carrier_SIFT_verified','carrier_context_roundtrip_root_verified','independent_carrier_border_flow'))
                require_visible_word(frame,instance['word_quad'],partial)
            raw,ink_mask=compose_planar_text(frame,instance['source_quad'],asset,style,return_mask=True)
            alpha=np.minimum(alpha,ink_mask)
            patch,effective=compose_frame(frame,raw,alpha,protection)
            touched=effective!=0
            if np.any(union & touched):
                raise ValueError(f'Unresolved operation overlap at source frame {n}')
            result[touched]=patch[touched];union|=touched
            if np.any(touched):
                report['modified_instance_count']+=1
                if not report['modified_source_frames'] or report['modified_source_frames'][-1]!=n:
                    report['modified_source_frames'].append(n)
    return result,union


def preview(case,manifest,output,start,end):
    from production_effort import begin_preview, finish_preview, recover_preview
    from fractions import Fraction
    item = __import__('workflow').item_for(case, manifest['file_id'])
    from source_timeline import load_for_item
    timeline = load_for_item(item)
    fps = float(Fraction(timeline['fps']))
    frames = [round(start*fps), round(end*fps)]
    if not 0 <= frames[0] < frames[1] <= len(timeline['pts']):
        raise ValueError('Invalid preview source range')
    recovered = recover_preview(case, manifest, output, frames)
    if recovered is not None:
        return recovered
    trial = begin_preview(case, manifest, output, frames=frames)
    try:
        report = _preview(case, manifest, output, start, end)
        if trial:
            finish_preview(case, manifest['file_id'], trial['id'], report=str(
                Path(str(output)+'.json').resolve().relative_to(project.ROOT)))
            report['trial_id'] = trial['id']
        return report
    except Exception as exc:
        if trial:
            finish_preview(case, manifest['file_id'], trial['id'], error=exc)
        raise


def _preview(case,manifest,output,start,end):
    """Short source-time preview through the same compositor as the full export.

    This diagnostic has no delivery/acceptance side effects. Full exports keep
    the original audio and timeline; this silent excerpt is for visual trials.
    """
    item,timeline,loaded,artifacts=validate(case,manifest)
    fps=float(__import__('fractions').Fraction(timeline['fps']))
    lo,hi=round(start*fps),round(end*fps)
    if not 0<=lo<hi<=len(timeline['pts']):raise ValueError('Invalid preview source range')
    output=Path(output)
    if output.exists():raise ValueError('Preserve existing preview')
    output.parent.mkdir(parents=True,exist_ok=True)
    layers,reports=load_layers(loaded,timeline)
    cap=cv2.VideoCapture(str(project.project_path(item['source'])))
    width,height=int(cap.get(3)),int(cap.get(4));cap.set(cv2.CAP_PROP_POS_FRAMES,lo)
    command=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
        '-r',str(timeline['fps']),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','16',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for n in range(lo,hi):
            ok,frame=cap.read()
            if not ok:raise ValueError('Source preview decode ended early')
            result,_=compose_layers(frame,n,layers)
            proc.stdin.write(result.tobytes())
        proc.stdin.close();error=proc.stderr.read().decode()
        if proc.wait(timeout=60):raise ValueError('Preview encode failed: '+error[-500:])
    finally:
        cap.release()
        for layer in layers:
            if hasattr(layer,'close'):layer.close()
        if proc.poll() is None:proc.kill();proc.wait()
        if not proc.stdin.closed:proc.stdin.close()
        proc.stderr.close()
    report=dict(file_id=item['id'],source_sha256=item['source_sha256'],source_frames_half_open=[lo,hi],
        artifact_hashes=artifacts,output=str(output.relative_to(project.ROOT)),output_sha256=project.digest(output),
        operations=reports,status='silent_segment_preview_pending_visual_review',accepted=False)
    Path(str(output)+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def render(case, manifest, output, diagnostic=False):
    from production_effort import require_preview, require_reviewed_repair
    require_reviewed_repair(case, manifest['file_id'], [e['operation_id'] for e in manifest['operations']])
    if not diagnostic:
        require_preview(case, manifest)
    from regional_composite import stream_hash
    import workflow
    project.validate_case(case)
    item, timeline, loaded, artifacts = validate(case, manifest)
    if any(p and p.get('composition_mode') == 'source_recolor' for _, p in loaded):
        from production_effort import require_lightweight_edit
        require_lightweight_edit(case)
    if diagnostic:
        ready = [p for _, p in loaded if p is not None]
        if not ready or any(p.get('composition_mode') != 'aigc_regional' or not p.get('diagnostic') for p in ready):
            raise ValueError('Diagnostic export requires explicitly diagnostic provider-pixel plans')
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; preserve parent')
    output.parent.mkdir(parents=True, exist_ok=True)
    layers,reports=load_layers(loaded,timeline)
    source = project.project_path(item['source'])
    cap = cv2.VideoCapture(str(source))
    width,height = int(cap.get(3)),int(cap.get(4))
    from fractions import Fraction
    fps = timeline['fps']
    track_timescale = str(Fraction(timeline['time_base']).denominator)
    command = ['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
               '-r',str(fps),'-i','-','-i',str(source),'-map','0:v:0','-map','1:a:0?',
               '-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-c:a','copy',
               '-video_track_timescale',track_timescale,'-movie_timescale','1000000','-movflags','+faststart',str(output)]
    proc = subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    modified=[]; changes=0; n=0
    try:
        while True:
            ok, frame=cap.read()
            if not ok: break
            result,union=compose_layers(frame,n,layers)
            diff=np.any(result!=frame,axis=2)
            if np.any(diff & ~union): raise ValueError('Write outside operation union')
            changes+=int(diff.sum())
            if np.any(union): modified.append(n)
            proc.stdin.write(result.tobytes()); n+=1
        proc.stdin.close();error=proc.stderr.read().decode();proc.stderr.close()
        if proc.wait(timeout=60) or n!=len(timeline['pts']): raise ValueError('Unified encode failed: '+error[-500:])
    finally:
        cap.release()
        for layer in layers:
            if hasattr(layer,'close'):layer.close()
        if proc.poll() is None: proc.kill();proc.wait()
        if not proc.stdin.closed: proc.stdin.close()
        if not proc.stderr.closed: proc.stderr.close()
    from source_timeline import build
    out_timeline=build(output)
    source_video=stream_hash('ffmpeg',source);output_video=stream_hash('ffmpeg',output)
    audio=project.probe(source)['audio_present']; out_audio=project.probe(output)['audio_present']
    a=stream_hash('ffmpeg',source,True) if audio else []
    b=stream_hash('ffmpeg',output,True) if out_audio else []
    # Compare rational presentation times, not ticks with possibly different time bases.
    from fractions import Fraction
    pts=lambda t:[Fraction(f)*Fraction(t['time_base']) for f in t['pts']]
    exact=pts(timeline)==pts(out_timeline)
    report={'file_id':item['id'],'case_version':case['case_version'],'source_sha256':item['source_sha256'],
            'dependency':workflow.requirement_binding(case,item),'output':str(output.relative_to(project.ROOT)),
            'output_sha256':project.digest(output),'parent_version':manifest.get('parent_version'),
            'source_frame_count':len(source_video),'output_frame_count':len(output_video),
            'source_frames_half_open':[0,n],'modified_source_frames':modified,'operations':reports,
            'timeline_exact':exact,'audio_packets_pts_payload_exact':audio==out_audio and a==b,
            'source_audio_packets':len(a),'output_audio_packets':len(b),'preencode_changes_outside_effective_mask':0,
            'preencode_changed_pixels':changes,'artifact_hashes':artifacts,'full_target_coverage':False,
            'accepted':False,'status':'full_length_candidate_pending_visual_review'}
    if not exact or not report['audio_packets_pts_payload_exact']: report['status']='failed_integrity_check'
    return report
