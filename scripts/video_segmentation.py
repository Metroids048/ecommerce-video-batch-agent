"""SAM2 video masks through the existing provider workflow, not polygon animation."""
import json
import cv2
import numpy as np
import project
import workflow
import runninghub_adapter as provider


MODEL_ID = 'sam2-video-base-plus'
LARGE_MODEL_ID = 'sam2-video-large-multiobject'
WORKFLOW_ID = '1864499239568211970'
SAM3_MODEL_ID = 'sam3-video-semantic'
DENSE_MODEL_ID = 'sam3-video-dense-components'


def dense_parameters(spec, dimensions, fps, frame_count):
    if set(spec) != {'width','height','parts'} or [spec['width'],spec['height']] != dimensions:
        raise ValueError('Dense component masks require the exact source grid')
    parts=spec['parts']
    if not isinstance(parts,list) or not 1 <= len(parts) <= 8:
        raise ValueError('One visible seed per distinct component is required')
    for i,p in enumerate(parts):
        if set(p) != {'name','seed_frame','frames_half_open','atlas_index'}:
            raise ValueError('Invalid dense component binding')
        lo,hi=p['frames_half_open'];seed=p['seed_frame']
        if (any(type(n) is not int for n in (lo,hi,seed,p['atlas_index'])) or
                not 0 <= lo <= seed < hi <= frame_count or p['atlas_index'] != i):
            raise ValueError('Dense component seed/range must bind exact source frames')
    return dict(spec,workflowId='2048031544987623426',instanceType='plus',fps=float(fps))


def dense_graph(p):
    nodes={'1':dict(class_type='VHS_LoadVideo',inputs=dict(video='__UPLOADED_VIDEO__',
        custom_width=p['width'],custom_height=p['height'],force_rate=0,frame_load_cap=0,
        skip_first_frames=0,select_every_nth=1,force_size='Disabled',format='None')),
        '2':dict(class_type='LoadImage',inputs=dict(image='__UPLOADED_SEED_ATLAS__')),
        '3':dict(class_type='CheckpointLoaderSimple',inputs=dict(ckpt_name='sam3.1_multiplex_fp16.safetensors'))}
    def add(kind,**inputs):
        key=str(len(nodes)+1);nodes[key]=dict(class_type=kind,inputs=inputs);return [key,0]
    def select(images,indices):
        return add('VHS_SelectImages',image=images,indexes=','.join(map(str,indices)),err_if_missing=True,err_if_empty=True)
    assembled=None
    for part in p['parts']:
        seed=add('ImageCrop',image=['2',0],width=p['width'],height=p['height'],x=part['atlas_index']*p['width'],y=0)
        seed=add('ImageToMask',image=seed,channel='red')
        lo,hi=part['frames_half_open'];anchor=part['seed_frame'];before=None
        for reverse,indices in [(True,list(range(anchor,lo-1,-1))),(False,list(range(anchor,hi)))]:
            if reverse and anchor == lo:continue
            images=select(['1',0],indices)
            track=add('SAM3_VideoTrack',images=images,model=['3',0],initial_mask=seed,
                detection_threshold=.5,max_objects=1,detect_interval=1)
            masks=add('SAM3_TrackToMask',track_data=track,object_indices='0')
            result=add('MaskToImage',mask=masks)
            if reverse:
                # Exclude the duplicated seed, restore original source order.
                before=select(result,range(anchor-lo,0,-1))
            else:
                result=add('ImageBatch',image1=before,image2=result) if before else result
                assembled=add('ImageBatch',image1=assembled,image2=result) if assembled else result
    add('VHS_VideoCombine',images=assembled,frame_rate=p['fps'],loop_count=0,
        filename_prefix='sam3_dense_masks',format='video/h264-mp4',pingpong=False,
        save_output=True,crf=0,pix_fmt='yuv420p',no_preview=False)
    return nodes


def unpack_dense_masks(masks,p,frame_count):
    expected=sum(b-a for a,b in (part['frames_half_open'] for part in p['parts']))
    if len(masks) != expected or not masks or any(m.shape != (p['height'],p['width']) for m in masks):
        raise ValueError('Dense mask stack incomplete; no missing-frame padding or retiming')
    alpha=np.zeros((frame_count,p['height'],p['width']),np.uint8);offset=0
    for part in p['parts']:
        lo,hi=part['frames_half_open'];n=hi-lo
        alpha[lo:hi]=np.maximum(alpha[lo:hi],np.stack(masks[offset:offset+n]));offset+=n
    return alpha


def fixed_recolor_parameters(item, operation):
    from math import gcd
    w, h = item['dimensions']; divisor = gcd(w, h)
    w, h = w // divisor, h // divisor
    # SAM3 consumes images, not VAE latents. Keep existing grids where possible;
    # otherwise use the even grid required by the lossless h264 mask output.
    alignment = 8 if 8 * max(w, h) <= 1024 else 2
    multiplier = 1024 // (alignment * max(w, h))
    if multiplier < 1 or min(w, h) * alignment * multiplier < 128:
        raise ValueError('Source aspect ratio has no supported SAM3 grid; do not distort or try alternate grids')
    return dict(width=w*alignment*multiplier, height=h*alignment*multiplier, text_prompt=operation['semantic_label'])


def sam3_parameters(spec, dimensions, fps):
    if set(spec) not in ({'width','height','text_prompt'}, {'width','height','text_prompt','anchor_frame'},
                        {'width','height','text_prompt','anchor_frame','point_prompt'}):
        raise ValueError('SAM3 requires an explicit source grid and object text')
    w,h=spec['width'],spec['height']
    if (any(type(n) is not int or not 128 <= n <= 1024 or n % 2 for n in (w,h))
            or w*dimensions[1] != h*dimensions[0]
            or not isinstance(spec['text_prompt'],str) or not 1 <= len(spec['text_prompt']) <= 200):
        raise ValueError('Invalid SAM3 source grid/object prompt')
    if 'anchor_frame' in spec and (type(spec['anchor_frame']) is not int or spec['anchor_frame'] < 0):
        raise ValueError('Invalid SAM3 visible anchor frame')
    if 'point_prompt' in spec:
        points = spec['point_prompt']
        if (set(points) != {'positive','negative'} or not 1 <= len(points['positive']) <= 16
                or len(points['negative']) > 16):
            raise ValueError('SAM3 instance prompt requires sparse positive/negative points')
        for point in points['positive'] + points['negative']:
            if (set(point) != {'x','y'} or any(type(point[k]) not in (int,float)
                    or not np.isfinite(point[k]) or not 0 <= point[k] < limit
                    for k,limit in [('x',w),('y',h)])):
                raise ValueError('SAM3 point outside its source grid')
    return dict(spec,workflowId='1992595203001479169',instanceType='plus',fps=float(fps))


def sam3_graph(p):
    # Current Easy-SAM3 video node, from the maintained public workflow.
    graph = {
        '1':dict(class_type='VHS_LoadVideo',inputs=dict(video='__UPLOADED_VIDEO__',
            custom_width=p['width'],custom_height=p['height'],force_rate=0,frame_load_cap=0,
            skip_first_frames=0,select_every_nth=1,force_size='Disabled',format='None')),
        '2':dict(class_type='easy sam3ModelLoader',inputs=dict(model='sam3.pt',
            segmentor='video',device='cuda',precision='fp16')),
        '3':dict(class_type='easy sam3VideoSegmentation',inputs=dict(
            sam3_model=['2',0],video_frames=['1',0],prompt=p['text_prompt'],object_id=1,
            frame_index=p.get('anchor_frame',0),score_threshold_detection=.5,new_det_thresh=.7,
            propagation_direction='both',start_frame_index=p.get('anchor_frame',0),max_frames_to_track=-1,
            keep_model_loaded=False,close_after_propagation=True)),
        '4':dict(class_type='MaskToImage',inputs=dict(mask=['3',0])),
        '69':dict(class_type='VHS_VideoCombine',inputs=dict(images=['4',0],frame_rate=p['fps'],
            loop_count=0,filename_prefix='sam3_masks',format='video/h264-mp4',pingpong=False,
            save_output=True,crf=0,pix_fmt='yuv420p',no_preview=False))}
    if 'point_prompt' in p:
        # SAM3 instance points and semantic text are mutually exclusive upstream.
        graph['3']['inputs'].update(prompt='',
            positive_coords=json.dumps(p['point_prompt']['positive']),
            negative_coords=json.dumps(p['point_prompt']['negative']))
    return graph


def parameters(spec, dimensions, fps, multiobject=False):
    if set(spec) != {'anchors', 'width', 'height'}:
        raise ValueError('SAM2 requires an explicit grid and sparse keyframe prompts')
    w, h = spec['width'], spec['height']
    if (any(type(n) is not int or n < 128 or n > 1024 or n % 2 for n in (w,h)) or
            w*dimensions[1] != h*dimensions[0]):
        raise ValueError('SAM2 grid must preserve the source aspect ratio')
    anchors = spec['anchors']
    if not isinstance(anchors,list) or not 1 <= len(anchors) <= 8 or anchors[0]['frame'] != 0:
        raise ValueError('Use at most eight sparse keyframes, beginning at the first source frame')
    previous = (-1, -1)
    for anchor in anchors:
        expected = {'frame','positive_points','negative_points'} | ({'object_id'} if multiobject else set())
        key = (anchor['frame'], anchor.get('object_id',1))
        if (set(anchor) != expected or type(anchor['frame']) is not int or key <= previous or
                type(key[1]) is not int or not 1 <= key[1] <= 4):
            raise ValueError('SAM2 anchor frames must be distinct and ordered')
        previous = key
        for key in ('positive_points', 'negative_points'):
            points = anchor[key]
            if not isinstance(points, list) or not (1 if key == 'positive_points' else 0) <= len(points) <= 16:
                raise ValueError('Use sparse anchor prompts, never per-frame corrections')
            if any(set(p) != {'x','y'} or any(type(p[k]) not in (int,float) or not np.isfinite(p[k])
                    or not 0 <= p[k] < limit for k,limit in [('x',w),('y',h)]) for p in points):
                raise ValueError('SAM2 point outside its explicit coordinate grid')
    return dict(spec, workflowId=WORKFLOW_ID, instanceType='default', fps=float(fps), multiobject=multiobject)


def graph(p):
    """Public SAM2 video predictor with sparse conditioning frames, one propagation."""
    nodes = {
        '1': dict(class_type='VHS_LoadVideo', inputs=dict(video='__UPLOADED_VIDEO__',
            custom_width=p['width'], custom_height=p['height'], force_rate=0,
            frame_load_cap=0, skip_first_frames=0, select_every_nth=1, force_size='Disabled',format='None')),
        '2': dict(class_type='DownloadAndLoadSAM2Model', inputs=dict(model='sam2_hiera_large.safetensors' if p.get('multiobject') else 'sam2_hiera_base_plus.safetensors',
            precision='bf16', device='cuda', segmentor='video'))}
    prior_model = ['2',0]
    prior_state = None
    for i,anchor in enumerate(p['anchors'],3):
        inputs = dict(sam2_model=prior_model, frame_index=anchor['frame'],object_index=anchor.get('object_id',1),
            coordinates_positive=json.dumps(anchor['positive_points']),
            coordinates_negative=json.dumps(anchor['negative_points']))
        inputs.update({'image':['1',0]} if prior_state is None else {'prev_inference_state':prior_state})
        nodes[str(i)] = dict(class_type='Sam2VideoSegmentationAddPoints',inputs=inputs)
        prior_model,prior_state = [str(i),0],[str(i),1]
    i = str(len(p['anchors'])+3)
    nodes[i] = dict(class_type='Sam2VideoSegmentation',inputs=dict(sam2_model=prior_model,
        inference_state=prior_state,keep_model_loaded=False))
    j = str(int(i)+1)
    nodes[j] = dict(class_type='MaskToImage',inputs=dict(mask=[i,0]))
    nodes[str(int(j)+1)] = dict(class_type='VHS_VideoCombine', inputs=dict(images=[j,0], frame_rate=p['fps'],
        loop_count=0, filename_prefix='sam2_masks', format='video/h264-mp4',
        pingpong=False, save_output=True, crf=0, pix_fmt='yuv420p',no_preview=False))
    return nodes


def import_masks(case, spec, out):
    item = workflow.item_for(case, spec['file_id'])
    state = project.read_json(workflow.state_path(case))
    workflow.verify_state_case(case, state)
    unit = provider.unit_for(state, item['id'], spec['unit_id'])
    request = provider.verify_unit(case, unit)
    if request['model_id'] not in (MODEL_ID,LARGE_MODEL_ID,SAM3_MODEL_ID,DENSE_MODEL_ID,'vitmatte-video-local') or unit['status'] != 'raw_ready':
        raise ValueError('Import only a completed source-bound SAM2 video task')
    raw = unit['raw']; path = project.project_path(raw['path'])
    if project.digest(path) != raw['sha256']:
        raise ValueError('SAM2 result changed')
    lo, hi = request['source_frames_half_open']
    output = project.project_path(out)
    sidecar = output.with_suffix('.npz.json')
    if output.suffix != '.npz':
        raise ValueError('Use a .npz mask artifact')
    if output.exists() or sidecar.exists():
        if not output.is_file() or not sidecar.is_file():
            raise ValueError('Preserve incomplete source-bound mask import; reconcile original artifacts')
        report = project.read_json(sidecar)
        if (report.get('file_id')!=item['id'] or report.get('unit_id')!=unit['unit_id'] or
                report.get('operation_id')!=unit['operation_id']):
            raise ValueError('Existing mask is not this source-bound video segmentation result')
        validate_import(case,item,dict(operation_id=unit['operation_id'],segmentation_import=str(sidecar.relative_to(project.ROOT)),
            segmentation_import_sha256=project.digest(sidecar),segmentation_backend=request['model_id'],
            masks=out,edit_frames_half_open=[lo,hi]))
        return dict(report,execution_reused=True)
    from production_pipeline import guard_upstream
    guard_upstream(case,item['id'],operation_id=unit['operation_id'])
    cap = cv2.VideoCapture(str(path)); masks = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masks.append(gray if request['model_id']=='vitmatte-video-local' else (gray >= 128).astype(np.uint8)*255)
    finally:
        cap.release()
    p = request['parameters']
    if request['model_id'] == DENSE_MODEL_ID:
        alpha=unpack_dense_masks(masks,p,hi-lo)
    elif len(masks) != hi-lo or not masks or masks[0].shape != (p['height'],p['width']):
        raise ValueError('SAM2 output must contain exactly one mask per source frame; no padding/retiming')
    else:
        alpha=np.stack(masks)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, alpha=alpha, protected=np.zeros_like(alpha),
            source_sha256=np.array(item['source_sha256']),source_dimensions=item['dimensions'],
            mask_dimensions=[p['width'],p['height']],frames_half_open=[lo,hi])
    report = dict(file_id=item['id'],operation_id=unit['operation_id'],unit_id=unit['unit_id'],
        request_fingerprint=unit['request_fingerprint'],raw_sha256=raw['sha256'],
        source_sha256=item['source_sha256'],masks=out,masks_sha256=project.digest(output),
        frames_half_open=[lo,hi],backend=request['model_id'],status='prepared_pending_visual_review',accepted=False)
    output.with_suffix('.npz.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def validate_import(case, item, plan):
    report_path = project.project_path(plan['segmentation_import'])
    if project.digest(report_path) != plan['segmentation_import_sha256']:
        raise ValueError('Segmentation import evidence changed')
    report = project.read_json(report_path)
    state = project.read_json(workflow.state_path(case)); workflow.verify_state_case(case,state)
    unit = provider.unit_for(state,item['id'],report['unit_id']); request = provider.verify_unit(case,unit)
    if (request['model_id'] not in (MODEL_ID,LARGE_MODEL_ID,SAM3_MODEL_ID,DENSE_MODEL_ID,'vitmatte-video-local') or report['backend'] != request['model_id'] or
            report.get('file_id') != item['id'] or report.get('operation_id') != plan['operation_id'] or
            report['backend'] != plan.get('segmentation_backend') or unit['status'] != 'raw_ready' or
            request['operation_id'] != plan['operation_id'] or report['source_sha256'] != item['source_sha256'] or
            unit['raw']['sha256'] != report['raw_sha256'] or unit['request_fingerprint'] != report['request_fingerprint'] or
            report['masks'] != plan['masks'] or report['masks_sha256'] != project.digest(project.project_path(plan['masks'])) or
            report['frames_half_open'] != plan['edit_frames_half_open']):
        raise ValueError('Recolor mask is not the recorded source-bound video segmentation result')
