"""Learned edge refinement of cached semantic masks; no second SAM propagation."""
import json
from fractions import Fraction
import subprocess
import cv2
import numpy as np
import project
import workflow

MODEL_ID = 'vitmatte-video-local'


def require_trimap_foreground(mask, edge_erode):
    # Match LayerStyle generate_VITMatte_trimap: kernel width, five iterations.
    known = np.where(mask > 127, 255, 0).astype(np.uint8)
    count,labels=cv2.connectedComponents(known)
    eroded=cv2.erode(known,np.ones((edge_erode,edge_erode),np.uint8),iterations=5)
    retained=set(np.unique(labels[eroded!=0]))
    if set(range(1,count))-retained:
        raise ValueError('Matting trimap loses confident foreground in a target component; erosion destroys thin details')


def parameters(case, item, spec, fps, interval):
    from source_recolor import validate
    p = spec['parameters']
    if set(p) != {'root_plan','width','height','edge_erode','edge_dilate'}:
        raise ValueError('Explicit source-bound matting input required')
    root = project.read_json(project.project_path(p['root_plan']))
    if root.get('alpha_refinement') or root['operation_id'] != spec['operation_id']:
        raise ValueError('Matte the semantic root, never a refined candidate')
    dependencies = validate(case,item,root)
    sidecar = project.read_json(project.project_path(spec['mask_video']+'.json'))
    expected = dict(source_sha256=item['source_sha256'],root_plan=p['root_plan'],
        root_plan_sha256=project.digest(project.project_path(p['root_plan'])),
        frames_half_open=interval,mask_video_sha256=project.digest(project.project_path(spec['mask_video'])),
        implementation_sha256=project.digest(project.ROOT/'scripts/source_recolor.py'))
    if any(sidecar.get(k) != v for k,v in expected.items()):
        raise ValueError('Matting mask video source/root/implementation changed')
    w,h=p['width'],p['height']
    if (type(w)is not int or type(h)is not int or not 128<=w<=1280 or w*item['dimensions'][1]!=h*item['dimensions'][0]
            or any(type(p[k])is not int or not 1<=p[k]<=64 for k in ('edge_erode','edge_dilate'))):
        raise ValueError('Invalid matting grid or trimap band')
    from source_timeline import build, require_cfr
    mt=require_cfr(build(project.project_path(spec['mask_video'])))
    if len(mt['pts'])!=interval[1]-interval[0] or Fraction(mt['fps'])!=Fraction(fps):
        raise ValueError('Matting input must retain every source frame')
    cap=cv2.VideoCapture(str(project.project_path(spec['mask_video'])))
    try:
        for frame in range(interval[1]-interval[0]):
            ok,image=cap.read()
            if not ok:raise ValueError('Matting mask ended before source')
            require_trimap_foreground(image[:,:,2],p['edge_erode'])
    finally:cap.release()
    dependencies[p['root_plan']]=expected['root_plan_sha256']
    dependencies[spec['mask_video']+'.json']=project.digest(project.project_path(spec['mask_video']+'.json'))
    return dict(p,workflowId='1864499239568211970',instanceType='default',fps=float(Fraction(fps)),
        workflow=graph(p,float(Fraction(fps)))),dependencies


def graph(p,fps):
    loader=dict(custom_width=p['width'],custom_height=p['height'],force_rate=0,frame_load_cap=0,
        skip_first_frames=0,select_every_nth=1,force_size='Disabled')
    return {
        '1':dict(class_type='VHS_LoadVideo',inputs=dict(loader,video='__UPLOADED_VIDEO__')),
        '2':dict(class_type='VHS_LoadVideo',inputs=dict(loader,video='__UPLOADED_MASK__')),
        '3':dict(class_type='ImageToMask',inputs=dict(image=['2',0],channel='red')),
        # LayerStyle casts to uint8 before scaling: decoded 253/255 would
        # otherwise become zero. Normalize exactly as the local preflight does.
        '7':dict(class_type='ThresholdMask',inputs=dict(mask=['3',0],value=.5)),
        '4':dict(class_type='LayerMask: MaskEdgeUltraDetail V2',inputs=dict(image=['1',0],mask=['7',0],
            method='VITMatte(local)',mask_grow=0,fix_gap=0,fix_threshold=.75,edge_erode=p['edge_erode'],
            edte_dilate=p['edge_dilate'],black_point=.01,white_point=.99,device='cuda',max_megapixels=1.)),
        '5':dict(class_type='MaskToImage',inputs=dict(mask=['4',1])),
        '6':dict(class_type='VHS_VideoCombine',inputs=dict(images=['5',0],frame_rate=fps,loop_count=0,
            filename_prefix='vitmatte_alpha',format='video/h264-mp4',pingpong=False,save_output=True,crf=0,pix_fmt='yuv420p'))}


def prepare_input(case,spec,out):
    from source_recolor import Layer,validate
    item=workflow.item_for(case,spec['file_id']);root=project.read_json(project.project_path(spec['root_plan']))
    validate(case,item,root)
    if root.get('alpha_refinement'):
        raise ValueError('Use cached semantic root')
    lo,hi=spec['frames_half_open'];w,h=spec['size']
    if not root['edit_frames_half_open'][0]<=lo<hi<=root['edit_frames_half_open'][1]:
        raise ValueError('Matting interval exceeds semantic root')
    target=project.project_path(out)
    if target.exists():raise ValueError('Preserve previous matting input')
    layer=Layer(root,dict(modified_source_frames=[],modified_instance_count=0))
    cap=cv2.VideoCapture(str(project.project_path(item['source'])));cap.set(1,lo)
    from source_timeline import load_for_item,require_cfr
    timeline=require_cfr(load_for_item(item))
    proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','gray','-s',f'{w}x{h}',
        '-r',timeline['fps'],'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','0','-pix_fmt','yuv420p',str(target)],stdin=subprocess.PIPE)
    try:
        for n in range(lo,hi):
            ok,frame=cap.read()
            if not ok:raise ValueError('Matting source ended early')
            proc.stdin.write(cv2.resize(layer.mask(frame,n),(w,h),interpolation=cv2.INTER_NEAREST).tobytes())
    finally:cap.release();proc.stdin.close()
    if proc.wait():raise ValueError('Matting input encoder failed')
    report=dict(source_sha256=item['source_sha256'],root_plan=spec['root_plan'],
        root_plan_sha256=project.digest(project.project_path(spec['root_plan'])),frames_half_open=[lo,hi],
        mask_video_sha256=project.digest(target),implementation_sha256=project.digest(project.ROOT/'scripts/source_recolor.py'))
    target.with_suffix(target.suffix+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
