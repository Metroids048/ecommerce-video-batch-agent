"""On-device text instances; source-bound observations, not an absence oracle."""
import json
import subprocess
import unicodedata
from pathlib import Path
import cv2
import numpy as np
import project


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',text.upper()) if c.isascii() and c.isalnum())


def old_print_findings(detections, record):
    found=[]
    for detection in detections:
        q=np.float32(detection['word_quad'])
        for instance in record.get('instances',[record]):
            points=instance.get('word_quad',instance.get('source_quad'))
            if points is None:continue
            overlap,_=cv2.intersectConvexConvex(cv2.convexHull(q),cv2.convexHull(np.float32(points)))
            if overlap/max(1,abs(cv2.contourArea(q)))>.5:
                found.append(detection);break
    return found


def audit_output(case, spec, out):
    """Inspect the actual encoded candidate; negative OCR never grants a pass."""
    import workflow
    from source_timeline import load_for_item
    item=workflow.item_for(case,spec['file_id'])
    state=workflow.snapshot(case)
    entry=state['files'][item['id']]
    candidate=workflow.verify_candidate(case,item['id'],
        workflow.candidate_for(entry,spec['candidate_id']))
    plan_path=project.project_path(spec['plan'])
    if candidate['artifact_hashes'].get(spec['plan'])!=project.digest(plan_path):
        raise ValueError('Text audit plan is not bound to this candidate')
    plan=project.read_json(plan_path)
    if plan['operation_id']!=spec['operation_id'] or plan['source_sha256']!=item['source_sha256']:
        raise ValueError('Text audit belongs to another operation/source')
    path=project.project_path(spec['scan'])
    rows=([json.loads(line)for line in path.read_text().splitlines()] if path.exists() else
          scan_source({'source':candidate['output'],'source_sha256':candidate['output_sha256']},
                      [spec['text']],spec['scan']))
    count=len(load_for_item(item)['pts'])
    from execution_plan import coverage_records
    records=coverage_records(plan,count)
    if (rows[0]['source_sha256']!=candidate['output_sha256'] or not rows[-1].get('complete')
            or rows[-1]['stride']!=1 or rows[-1]['ocr_frames']!=count
            or rows[-1]['decoded_frames']!=count
            or [r['frame']for r in rows[1:-1]]!=list(range(count))):
        raise ValueError('Output audit requires a complete same-candidate text scan')
    findings=[];unassigned=[]
    for row,record in zip(rows[1:-1],records):
        detections=select_instances(row['passes'],spec['text'],row['width'],row['height'])
        bound=old_print_findings(detections,record)
        findings.extend(dict(d,frame=row['frame'])for d in bound)
        unassigned.extend(dict(d,frame=row['frame'])for d in detections if d not in bound)
    result={'file_id':item['id'],'candidate_id':candidate['id'],
            'source_sha256':item['source_sha256'],'output_sha256':candidate['output_sha256'],
            'dependency':candidate['dependency'],'operation_id':spec['operation_id'],
            'scan':spec['scan'],'scan_sha256':project.digest(path),
            'plan':spec['plan'],'plan_sha256':project.digest(plan_path),
            'decoded_frames':count,'old_print_on_target':findings,
            'unassigned_old_print_observations':unassigned,
            'status':'fail' if findings else 'reviewed_with_limits','accepted':False,
            'scope':'Actual encoded full-video OCR. Bound old print is a defect; unassigned print needs carrier identity review. No detections do not prove absence, coverage or quality.'}
    output=project.project_path(out)
    if output.exists():raise ValueError('Preserve existing output audit')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return {'audit':out,'status':result['status'],'bound_old_prints':len(findings),
            'unassigned_observations':len(unassigned),'accepted':False}


def select_instances(passes, target, width, height):
    target=normalized(target)
    found=[]
    for scan in passes:
        for row in scan['rows']:
            word=normalized(row['text'])
            # Never authorize a cropped partial word by a substring match.
            if (len(word)!=len(target) or row['confidence']<.8 or
                    sum(a!=b for a,b in zip(word,target)) > (1 if len(target)>=4 else 0)):
                continue
            points=np.float32(row['quad'])
            if scan['orientation']=='up':
                quad=np.column_stack([points[:,0]*width,(1-points[:,1])*height])
            elif scan['orientation']=='rightMirrored':
                quad=np.column_stack([points[:,1]*width,(1-points[:,0])*height])
            else:
                raise ValueError('Unsupported native OCR orientation')
            area=abs(cv2.contourArea(quad))
            if area<50 or not np.isfinite(quad).all(): continue
            duplicate=False
            for prior in found:
                other=np.float32(prior['word_quad'])
                overlap,_=cv2.intersectConvexConvex(cv2.convexHull(quad),cv2.convexHull(other))
                if overlap/max(1,min(area,abs(cv2.contourArea(other))))>.5:
                    duplicate=True;break
            if not duplicate:
                found.append({'word_quad':quad.tolist(),'recognized_text':row['text'],
                              'text_exact':word==target,'confidence':row['confidence'],
                              'mirrored':scan['orientation']=='rightMirrored',
                              'track_method':'on_device_text_recognition'})
    return found


def scan_source(item, words, out, stride=1):
    """Freeze a real decoded scan. A partial/failed scan cannot become a plan."""
    out=project.project_path(out)
    if out.exists(): raise ValueError('Preserve existing OCR observations')
    code=project.ROOT/'scripts/local_text_scan.swift'
    binary=project.ROOT/'runs/.tools'/('text-scan-'+project.digest(code)[:16])
    if not binary.exists():
        binary.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['swiftc',str(code),'-o',str(binary)],check=True,capture_output=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as stream:
        stream.write(json.dumps({'source_sha256':item['source_sha256'],'scanner_sha256':project.digest(code),
                                 'words':words,'on_device':True})+'\n');stream.flush()
        subprocess.run([str(binary),str(project.project_path(item['source'])),'|'.join(words),str(stride)],
                       stdout=stream,check=True)
    rows=[json.loads(line) for line in out.read_text().splitlines()]
    if not rows[-1].get('complete'):
        raise ValueError('Incomplete source OCR scan')
    return rows


def expand_word(quad, horizontal=1.10, vertical=1.28):
    q=np.float32(quad)
    unit=np.float32([[0,0],[1,0],[1,1],[0,1]])
    margin=np.float32([[(1-horizontal)/2,(1-vertical)/2],[(1+horizontal)/2,(1-vertical)/2],
                       [(1+horizontal)/2,(1+vertical)/2],[(1-horizontal)/2,(1+vertical)/2]])
    return cv2.perspectiveTransform(margin[None],cv2.getPerspectiveTransform(unit,q))[0]


def match_reference(reference, reference_quad, frame, word_quad):
    """Verify the source-observed print, including polarity; same text is not identity.

    Rectifying OCR's reading order also normalizes mirrored carriers. Positive
    correlation rejects inverted print on a different product (e.g. a shade name).
    This is a conservative appearance check, not semantic carrier recognition.
    """
    plane=np.float32([[0,0],[191,0],[191,63],[0,63]])
    def patch(image, quad):
        p=cv2.warpPerspective(image,cv2.getPerspectiveTransform(np.float32(quad),plane),(192,64))
        return cv2.cvtColor(p,cv2.COLOR_BGR2GRAY).astype(np.float32)
    area=abs(cv2.contourArea(np.float32(word_quad)))
    reference_area=abs(cv2.contourArea(np.float32(reference_quad)))
    if area < reference_area*.01:return False
    a,b=patch(reference,reference_quad),patch(frame,word_quad)
    if min(a.std(),b.std())<3:return False
    # A finger can hide one end of the print. Require the remaining contiguous
    # half to match; polarity is retained, so inverse print is still rejected.
    a=cv2.GaussianBlur(a,(0,0),2);b=cv2.GaussianBlur(b,(0,0),2)
    scores=[np.corrcoef(a[:,left:right].ravel(),b[:,left:right].ravel())[0,1]
            for left,right in ((0,192),(0,128),(64,192))]
    return bool(max(scores)>.35)


def propagate(anchor, current, quad, max_error=1.5):
    """Direct anchor flow plus round-trip and print correlation, never accumulated."""
    q=np.float32(quad)
    mask=np.zeros(anchor.shape,np.uint8);cv2.fillConvexPoly(mask,np.int32(q),255)
    points=cv2.goodFeaturesToTrack(anchor,160,.004,3,mask=mask)
    if points is None or len(points)<8:return None
    moved,status,_=cv2.calcOpticalFlowPyrLK(anchor,current,points,None,winSize=(31,31),maxLevel=3)
    back,back_status,_=cv2.calcOpticalFlowPyrLK(current,anchor,moved,None,winSize=(31,31),maxLevel=3)
    good=(status.ravel()!=0)&(back_status.ravel()!=0)&(np.linalg.norm(points-back,axis=2).ravel()<max_error)
    if good.sum()<8:return None
    matrix,support=cv2.findHomography(points[good],moved[good],cv2.RANSAC,1.5)
    if matrix is None or support.sum()<8:return None
    projected=cv2.perspectiveTransform(q[None],matrix)[0]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):return None
    area=abs(cv2.contourArea(projected))/max(1,abs(cv2.contourArea(q)))
    if not .5<area<2:return None
    # Verify that the old print is still on this plane; a geometric fit alone
    # previously drifted off a cloth tag and wrote the asset onto subtitles.
    plane=np.float32([[0,0],[159,0],[159,47],[0,47]])
    ref=cv2.warpPerspective(anchor,cv2.getPerspectiveTransform(q,plane),(160,48))
    patch=cv2.warpPerspective(current,cv2.getPerspectiveTransform(projected,plane),(160,48))
    if ref.std()<3 or patch.std()<3 or np.corrcoef(ref.ravel(),patch.ravel())[0,1]<.80:return None
    return projected


def require_visible_word(frame, quad, projected_partial=False):
    word=np.float32(quad)
    height,width=frame.shape[:2]
    if projected_partial:
        if not np.isfinite(word).all() or not cv2.isContourConvex(word):
            raise ValueError('Invalid projected carrier')
        image=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
        area,_=cv2.intersectConvexConvex(word,image)
        if area/max(1,abs(cv2.contourArea(word)))>=.25:return
        raise ValueError('Too little visible carrier for partial-surface repair')
    if (not np.isfinite(word).all() or (word[:,0]<0).any() or
            (word[:,0]>width-1).any() or (word[:,1]<0).any() or
            (word[:,1]>height-1).any()):
        raise ValueError('Clipped source word requires a visible-surface/occlusion method')


def prepare_instance(frame, instance, asset, style):
    """Reacquire context around the complete word, without enlarging the logo.

    Stop when no context can separate the print from the boundary. This does
    not authorize painting through an unseparated occluder.
    """
    from surface_text import compose_planar_text
    partial=(style.get('surface_cleanup')=='texture_inpaint' and
             instance.get('track_method') in ('source_carrier_SIFT_verified','carrier_context_roundtrip_root_verified','independent_carrier_border_flow'))
    require_visible_word(frame,instance['word_quad'],partial)
    if style.get('source_print_profile') and 'extent_refinement' not in instance:
        from print_registration import register_extent,fit_word_blur
        extent=register_extent(style['source_print_profile'],frame,instance['word_quad'],style['polarity'])
        instance['extent_refinement']=dict(extent,proposal_quad=instance['word_quad'])
        if extent['supported']:instance['word_quad']=extent['word_quad']
        require_visible_word(frame,instance['word_quad'],partial)
    if style.get('source_print_profile'):
        from print_registration import fit_word_blur
        instance['source_psf']=fit_word_blur(style['source_print_profile'],frame,instance['word_quad'],style['polarity'])
    error=ValueError('Complete old print extent exceeds available cleanup context')
    for horizontal,vertical in ((1.10,1.28),(1.20,1.50),(1.30,1.80),(1.50,2.00)):
        quad=expand_word(instance['word_quad'],horizontal,vertical)
        local=dict(style,projected_partial=partial,asset_fit=[style['asset_fit'][0]*1.10/horizontal,
                                   style['asset_fit'][1]*1.28/vertical],
                   cleanup_word_box=[(1-1/horizontal)/2,(1-1/vertical)/2,
                                     (1+1/horizontal)/2,(1+1/vertical)/2])
        if instance.get('extent_refinement',{}).get('supported'):
            unit=np.float32([[0,0],[1,0],[1,1],[0,1]])
            observed=np.float32(instance['extent_refinement']['proposal_quad'])
            polygon=cv2.perspectiveTransform(observed[None],cv2.getPerspectiveTransform(quad,unit))[0]
            if (polygon<0).any() or (polygon>1).any():continue
            local['cleanup_observed_polygon']=polygon.tolist()
        psf=instance.get('source_psf')
        if psf and psf['score']>=.7:
            directional=psf['direction_supported']
            theta=np.deg2rad(psf['angle_degrees'] if directional else 0);c,sn=np.cos(theta),np.sin(theta)
            rotation=np.float64([[c,-sn],[sn,c]])
            major,minor=(psf['major_sigma'],psf['minor_sigma']) if directional else (psf['isotropic_sigma'],)*2
            covariance=rotation@np.diag([major**2,minor**2])@rotation.T
            rw,rh=style['source_print_profile']['size'];pw,ph=style['plane_size']
            scaling=np.diag([pw/horizontal/rw,ph/vertical/rh])
            covariance+=np.eye(2)*style.get('reference_print_blur_sigma',0.)**2
            local['print_blur_covariance']=(scaling@covariance@scaling).tolist()
        try:compose_planar_text(frame,quad,asset,local,return_mask=True)
        except ValueError as failure:error=failure;continue
        instance.update(source_quad=quad.tolist(),label_style=local)
        return
    raise error


def label_style(spec,reference):
    pw,ph=spec.get('plane_size',[512,160])
    style={'plane_size':[pw,ph],'polarity':spec.get('polarity','dark'),
           'asset_fit':spec.get('asset_fit',[.84,.65]),'paper_cleanup':'robust_border'}
    if spec.get('surface_cleanup'):
        from surface_text import source_ink_template
        style.update(surface_cleanup=spec['surface_cleanup'],
                     source_ink_template=source_ink_template(reference,spec['reference_quad'],[pw,ph],style['polarity']))
    if spec.get('register_full_print'):
        from print_registration import freeze_profile,profile_image
        from surface_text import measured_blur
        style['source_print_profile']=freeze_profile(reference,spec['reference_quad'])
        style['reference_print_blur_sigma']=measured_blur(cv2.cvtColor(profile_image(style['source_print_profile']),cv2.COLOR_BGR2GRAY),style['polarity'])
    return style


def reprepare(case,parent,spec,out):
    """Revalidate source-bound geometry after a compositor change, without OCR/flow retries."""
    import copy
    project.validate_case(case)
    path=project.project_path(spec['parent_plan'])
    if project.digest(path)!=spec['parent_plan_sha256'] or project.read_json(path)!=parent:
        raise ValueError('Geometry parent changed')
    item=next(f for f in case['files'] if f['id']==parent['file_id'])
    if (parent['case_version']!=case['case_version'] or
            parent['source_sha256']!=item['source_sha256'] or spec['source_sha256']!=item['source_sha256'] or
            parent['spec']['method']!='on_device_text'):
        raise ValueError('Geometry belongs to another source/request')
    for key in ('masks','exact_asset','localization_evidence'):
        if project.digest(project.project_path(parent[key]))!=parent[key+'_sha256']:
            raise ValueError('Geometry dependency changed: '+key)
    output=project.project_path(out)
    if output.exists():raise ValueError('Preserve existing surface plan')
    with np.load(project.project_path(parent['masks']),allow_pickle=False) as stored:
        if stored['protected'].any():raise ValueError('Explicit protection needs its own geometric remapping')
        alpha=np.zeros_like(stored['alpha'])
    plan=copy.deepcopy(parent);records=plan['review']['records']
    cap=cv2.VideoCapture(str(project.project_path(item['source'])))
    cap.set(cv2.CAP_PROP_POS_FRAMES,parent['spec']['reference_frame']);ok,reference=cap.read()
    if not ok:raise ValueError('Source reference unavailable')
    overrides=spec.get('style_overrides',{})
    if set(overrides)-{'register_full_print'}:raise ValueError('Unsupported surface style override')
    plan['spec'].update(overrides)
    style=label_style(plan['spec'],reference)
    if spec.get('carrier_scan'):
        from carrier_geometry import augment_records
        evidence=augment_records(item,records,style,dict(spec,reference_quad=parent['spec']['reference_quad']))
        plan.update(carrier_evidence=evidence['scan'],carrier_evidence_sha256=evidence['scan_sha256'])
        plan['review']['independent_carrier_search']=evidence
    asset=cv2.imread(str(project.project_path(parent['exact_asset'])),cv2.IMREAD_UNCHANGED)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    for n,record in enumerate(records):
        ok,frame=cap.read()
        if not ok:raise ValueError('Source preflight ended early')
        for instance in record['instances']:
            instance.pop('blocked_reason',None);instance['state']='visible'
            try:prepare_instance(frame,instance,asset,style)
            except ValueError as error:instance.update(state='unresolved',blocked_reason=str(error))
        visible=any(i['state']=='visible' for i in record['instances'])
        blocked=any(i['state']=='unresolved' for i in record['instances'])
        record['state']='partial' if visible and blocked else 'visible' if visible else 'unresolved'
        if visible:alpha[n]=255
    cap.release()
    masks=output.with_suffix('.npz');output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(masks,alpha=alpha,protected=np.zeros_like(alpha))
    plan.update(label_style=style,masks=str(masks.relative_to(project.ROOT)),masks_sha256=project.digest(masks),
                composition_implementation='scripts/surface_text.py',
                composition_implementation_sha256=project.digest(project.ROOT/'scripts/surface_text.py'),
                geometry_parent=spec['parent_plan'],geometry_parent_sha256=spec['parent_plan_sha256'])
    plan['review']['frame_states']=[r['state'] for r in records]
    plan['review']['full_mask_reviewed']=False
    output.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    return dict(plan=out,geometry_reused_from=spec['parent_plan'],localized_frames=int(np.any(alpha,axis=(1,2)).sum()),
                status='surface_reprepared_pending_actual_visual_review')


def localize(case, spec, out):
    project.validate_case(case)
    item=next(f for f in case['files'] if f['id']==spec['file_id'])
    if item['source_sha256']!=spec['source_sha256']:raise ValueError('Text scan belongs to another source')
    output=project.project_path(out)
    if output.exists():raise ValueError('Preserve existing localization version')
    scan_path=spec.get('scan') or str(output.with_suffix('.ocr.jsonl').relative_to(project.ROOT))
    path=project.project_path(scan_path)
    rows=([json.loads(line) for line in path.read_text().splitlines()] if path.exists()
          else scan_source(item,[spec['text']],scan_path))
    if rows[0]['source_sha256']!=item['source_sha256'] or not rows[-1].get('complete'):
        raise ValueError('Unbound or incomplete source text scan')
    from source_timeline import load_for_item
    count=len(load_for_item(item)['pts'])
    if rows[-1]['decoded_frames']!=count or rows[-1]['stride']!=1 or rows[-1]['ocr_frames']!=count:
        raise ValueError('Full localization requires OCR observations for every decoded frame')
    observations=rows[1:-1]
    if [r['frame'] for r in observations]!=list(range(count)):raise ValueError('Missing/duplicate source text observations')
    width,height=observations[0]['width'],observations[0]['height']
    records=[]
    for row in observations:
        instances=select_instances(row['passes'],spec['text'],width,height)
        for j,instance in enumerate(instances):
            instance.update(instance_id=f"ocr-{row['frame']}-{j}",state='visible')
        records.append({'frame':row['frame'],'instances':instances})
    reference_frame=spec['reference_frame']
    reference_quad=spec['reference_quad']
    cap=cv2.VideoCapture(str(project.project_path(item['source'])))
    cap.set(cv2.CAP_PROP_POS_FRAMES,reference_frame)
    ok,reference=cap.read();cap.release()
    if not ok:raise ValueError('Source reference frame is unavailable')
    # Bounded automatic gap recovery in both directions from actual OCR anchors.
    # Nothing is labelled absent just because recognition/flow failed.
    cap=cv2.VideoCapture(str(project.project_path(item['source'])))
    scale=min(1,720/width)
    grays=[]
    rejected_identity=[]
    n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        valid=[]
        for instance in records[n]['instances']:
            if match_reference(reference,reference_quad,frame,instance['word_quad']):valid.append(instance)
            else:rejected_identity.append(dict(instance,frame=n,reason='different_source_print_appearance'))
        records[n]['instances']=valid
        if spec.get('relocalize_carrier') and n%12==0 and not valid:
            from carrier_tracking import relocate
            for j,instance in enumerate(relocate(reference,reference_quad,frame)):
                instance.update(instance_id=f'carrier-{n}-{j}')
                records[n]['instances'].append(instance)
        grays.append(cv2.cvtColor(cv2.resize(frame,None,fx=scale,fy=scale),cv2.COLOR_BGR2GRAY))
        n+=1
    cap.release()
    if len(grays)!=count:raise ValueError('Source decode changed during localization')
    anchors=[(r['frame'],dict(i)) for r in records for i in r['instances']]
    context=None
    if spec.get('tracking')=='carrier_context':
        from carrier_tracking import context_candidates
        context=context_candidates(grays,records,scale)

    for seed,anchor in ([] if context is not None else anchors):
        for direction in (-1,1):
            for distance in range(1,31):
                n=seed+direction*distance
                if not 0<=n<count:break
                q=propagate(grays[seed],grays[n],np.float32(anchor['word_quad'])*scale)
                if q is None:break
                q/=scale
                duplicates=[]
                for prior in records[n]['instances']:
                    overlap,_=cv2.intersectConvexConvex(cv2.convexHull(q),cv2.convexHull(np.float32(prior['word_quad'])))
                    if overlap/max(1,abs(cv2.contourArea(q)))>.35:duplicates.append(prior)
                if duplicates:break
                records[n]['instances'].append(dict(anchor,word_quad=q.tolist(),
                    instance_id=f"flow-{seed}-{anchor['instance_id']}-{n}",track_method='direct_OCR_anchor_verified_flow',anchor_frame=seed))
    pw,ph=spec.get('plane_size',[512,160])
    style=label_style(spec,reference)
    asset_path=next(op['replacement_asset'] for op in item['operations'] if op['id']==spec['operation_id'])
    asset=cv2.imread(str(project.project_path(asset_path)),cv2.IMREAD_UNCHANGED)
    cap=cv2.VideoCapture(str(project.project_path(item['source'])))
    alpha=np.zeros((count,ph,pw),np.uint8)
    rejected=0
    for n,record in enumerate(records):
        ok,frame=cap.read()
        if not ok:raise ValueError('Source preflight decode ended early')
        if context is not None:
            from planar_text import distinct_instances
            # Test independent roots before choosing, so a truncated OCR box
            # does not force a one-frame fallback to the original brand.
            alternatives=record['instances']+context[n]
            for candidate in alternatives:
                candidate['inliers']=candidate.get('root_appearance_score',1.)*100-min(30,candidate.get('root_distance',0))*.1
            ready=[]
            for candidate in sorted(alternatives,key=lambda i:i['inliers'],reverse=True):
                geometry=dict(candidate,source_quad=expand_word(candidate['word_quad'],1.10,1.28).tolist())
                if len(distinct_instances(ready+[geometry]))==len(ready):continue
                try:prepare_instance(frame,candidate,asset,style)
                except ValueError:continue
                ready.append(candidate)
            chosen=distinct_instances(ready)
            if chosen:record['instances']=chosen
        for instance in record['instances']:
            try:
                prepare_instance(frame,instance,asset,style)
            except ValueError as error:
                recovered=False
                # A recognition box/first flow hypothesis may truncate an end.
                # Reacquire from independent neighboring OCR anchors rather
                # than switching this one frame back to the old word.
                nearby=sorted(((seed,anchor) for seed,anchor in anchors if 0<abs(seed-n)<=30),
                              key=lambda pair:abs(pair[0]-n))
                for seed,anchor in nearby:
                    q=propagate(grays[seed],grays[n],np.float32(anchor['word_quad'])*scale)
                    if q is None:continue
                    q/=scale
                    old=np.float32(instance['word_quad'])
                    overlap,_=cv2.intersectConvexConvex(cv2.convexHull(q),cv2.convexHull(old))
                    if overlap/max(1,abs(cv2.contourArea(old)))<.5:continue
                    candidate=dict(instance,word_quad=q.tolist(),anchor_frame=seed,
                                   track_method='independent_anchor_cleanup_recovery')
                    try:prepare_instance(frame,candidate,asset,style)
                    except ValueError:continue
                    instance.update(candidate);recovered=True;break
                if not recovered:
                    instance.update(state='unresolved',blocked_reason=str(error));rejected+=1
        visible=any(i['state']=='visible' for i in record['instances'])
        blocked=any(i['state']=='unresolved' for i in record['instances'])
        record['state']='partial' if visible and blocked else 'visible' if visible else 'unresolved'
        if visible:alpha[n]=255
    cap.release()
    del grays
    masks=output.with_suffix('.npz');output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(masks,alpha=alpha,protected=np.zeros_like(alpha))
    plan={'case_version':case['case_version'],'file_id':item['id'],'operation_id':spec['operation_id'],
          'source_sha256':item['source_sha256'],'source_frame_count':count,'source_frames_half_open':[0,count],
          'cloud':item['source'],'cloud_sha256':item['source_sha256'],
          'masks':str(masks.relative_to(project.ROOT)),'masks_sha256':project.digest(masks),
          'mask_space':'label_plane','composition_mode':'exact_asset_on_source_label',
          'exact_asset':asset_path,'exact_asset_sha256':project.digest(project.project_path(asset_path)),
          'composition_implementation':'scripts/surface_text.py' if spec.get('surface_cleanup') else 'scripts/exact_asset.py',
          'composition_implementation_sha256':project.digest(project.ROOT/('scripts/surface_text.py' if spec.get('surface_cleanup') else 'scripts/exact_asset.py')),
          'localization_evidence':scan_path,'localization_evidence_sha256':project.digest(path),
          'label_style':style,'spec':spec,'parent_version':None,'provider_task_id':None,
          'full_scope_requested':True,'full_target_coverage':False,
          'localization_limit':'On-device OCR of all frames, all recognized instances; unreadable/occluded targets remain unresolved. No automatic absence proof.',
          'review':{'method':'all-frame on-device Vision OCR; '+('carrier-context root-verified registration' if context is not None else 'direct anchor verified short flow')+'; actual-source cleanup preflight',
                    'frame_states':[r['state'] for r in records],'records':records,
                    'rejected_identity':rejected_identity,'full_mask_reviewed':False}}
    output.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    return {'plan':out,'frames':count,'localized_frames':sum(bool(a.any()) for a in alpha),
            'instances':sum(sum(i['state']=='visible' for i in r['instances']) for r in records),
            'rejected_instances':rejected,'status':'pending_actual_visual_and_coverage_review'}
