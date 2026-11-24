"""Provider pixels as a method in the existing unified source-based compositor."""
from fractions import Fraction
import json
import cv2
import numpy as np
import project
import source_timeline
import workflow
import runninghub_adapter as provider


def region_context(state, item, parts, states, color_match):
    inputs=[]
    for part in parts:
        unit=provider.unit_for(state,item['id'],part['unit_id'])
        request=project.read_json(project.project_path(unit['request_path']))
        inputs.append(dict(unit_id=part['unit_id'],raw_sha256=unit['raw']['sha256'],
            masks_sha256=part['masks_sha256'],blend_mode=part.get('blend_mode','alpha'),
            **({'target_rgb':part['target_rgb']} if part.get('blend_mode')=='source_chroma' else {}),
            **({'timing_mapping':part['timing_mapping']} if part.get('timing_mapping') else {}),
            write_frames_half_open=part.get('write_frames_half_open',request['source_frames_half_open'])))
    return workflow.hash_json(dict(parts=inputs,frame_states=states,color_match=color_match,
        code={p:project.digest(project.project_path(p)) for p in
              ('scripts/aigc_composite.py','scripts/execution_plan.py')}))


def review_write_region(case, item, operation_id, unit, part, color_match, context):
    """Review only the pixels actually used, while retaining a failed raw verdict.

    This is a host visual observation, not an automatic quality score. Mask,
    mapping, blend, source range or donor changes invalidate it.
    """
    path = part.get('write_region_review')
    if not path:
        return {}
    review = project.read_json(project.project_path(path))
    request = provider.verify_unit(case, unit)
    expected = dict(file_id=item['id'], operation_id=operation_id,
        source_sha256=item['source_sha256'], raw_sha256=unit['raw']['sha256'],
        request_fingerprint=unit['request_fingerprint'], raw_review_sha256=unit['raw_review']['sha256'],
        masks_sha256=part['masks_sha256'], write_frames_half_open=part.get(
            'write_frames_half_open', request['source_frames_half_open']),
        blend_mode=part.get('blend_mode', 'alpha'), color_match=color_match,
        composition_context_sha256=context)
    if review.get('binding') != expected:
        raise ValueError('Write-region review binding mismatch')
    lo, hi = expected['write_frames_half_open']
    checks = review.get('checks', [])
    if (len(checks) != len(provider.RAW_CHECKS)
            or {c.get('kind') for c in checks} != provider.RAW_CHECKS
            or any(c.get('status') != 'pass' or not c.get('scope') for c in checks)
            or review.get('observed_source_frames') != list(range(lo, hi))
            or not review.get('evidence')):
        raise ValueError('Write-region review needs all written frames and six passing observations')
    artifacts = {path: project.digest(project.project_path(path))}
    for evidence in review['evidence']:
        if project.digest(project.project_path(evidence['path'])) != evidence['sha256']:
            raise ValueError('Write-region evidence changed')
        artifacts[evidence['path']] = evidence['sha256']
    return artifacts


def require_disjoint_parts(parts):
    """Allow disjoint masks or an explicit two-donor, same-frame crossfade."""
    for index,part in enumerate(parts):
        wl,wh=part['write_frames_half_open']
        for prior in parts[:index]:
            lo=max(wl,prior['write_frames_half_open'][0]);hi=min(wh,prior['write_frames_half_open'][1])
            if lo>=hi:continue
            if 'native_replace' in (part.get('blend_mode'), prior.get('blend_mode')):
                raise ValueError('Native replacement intervals must not overlap')
            masks=[]
            for p in (part,prior):
                with np.load(project.project_path(p['masks']),allow_pickle=False) as z:
                    offset=p['source_frames_half_open'][0]
                    effective=z['alpha'][lo-offset:hi-offset].copy()
                    if 'protected' in z:effective[z['protected'][lo-offset:hi-offset]>0]=0
                    masks.append(effective)
            if part.get('blend_mode')==prior.get('blend_mode') and part.get('blend_mode') in ('temporal_crossfade','source_chroma'):
                if np.any(masks[0].astype(np.uint16)+masks[1]>255):
                    raise ValueError('Crossfade alpha exceeds 255')
                if any(max(lo,p['write_frames_half_open'][0])<min(hi,p['write_frames_half_open'][1])
                       for p in parts[:index] if p is not prior):
                    raise ValueError('Crossfade supports at most two simultaneous donors')
            elif np.any((masks[0]>0)&(masks[1]>0)):
                raise ValueError('Overlapping provider pixels are unresolved')


def validate_frame_index_mapping(timeline, raw_timeline, interval, mapping, binding, reference_timeline=None):
    """Restore source PTS only for observed 1:1 frames and tiny nominal-rate rounding.

    This cannot repair missing frames, changed motion, or arbitrary speed changes.
    The host must observe correspondence; the program checks its exact binding,
    full frame inventory and a sub-half-frame total timestamp discrepancy.
    """
    lo, hi = interval
    if len(raw_timeline['pts']) != hi-lo:
        raise ValueError('Raw duration/frame rate differs; padding/retiming forbidden')
    if (mapping.get('method') != 'one_to_one_source_pts'
            or mapping.get('binding') != binding
            or mapping.get('observed_source_frames') != list(range(lo, hi))
            or not mapping.get('evidence')):
        raise ValueError('Frame correspondence audit is missing or bound to another source/raw')
    source_rate = Fraction(timeline['fps']); raw_rate = Fraction(raw_timeline['fps'])
    if reference_timeline is not None:
        # A declared input clock adapter changes timestamps before submission,
        # never frame order/count. A return must match that exact sent clock.
        if (len(reference_timeline['pts']) != hi-lo or
                mapping.get('reference_sha256') != reference_timeline['source_sha256'] or
                mapping.get('frame_correspondence') != 'one_to_one_reference_frames'):
            raise ValueError('Reference clock/frame correspondence audit is missing')
        source_rate = Fraction(reference_timeline['fps'])
        timeline = reference_timeline
        lo,hi = 0,hi-lo
    if abs(raw_rate/source_rate-1) > Fraction(1, 1000):
        raise ValueError('Rate difference exceeds nominal rounding; retiming forbidden')
    local = [Fraction(t-timeline['pts'][lo])*Fraction(timeline['time_base'])
             for t in timeline['pts'][lo:hi]]
    raw = [Fraction(t)*Fraction(raw_timeline['time_base']) for t in raw_timeline['pts']]
    if any(abs(a-b) >= 1/(2*source_rate) for a,b in zip(local,raw)):
        raise ValueError('Timestamp discrepancy reaches half a source frame; retiming forbidden')


def mapping_binding(item, unit, interval):
    return dict(source_sha256=item['source_sha256'], raw_sha256=unit['raw']['sha256'],
                request_fingerprint=unit['request_fingerprint'], source_frames_half_open=interval)


def prepare_plan(case, spec):
    state=project.read_json(workflow.state_path(case));workflow.verify_state_case(case,state)
    item=workflow.item_for(case,spec['file_id'])
    timeline=source_timeline.load_for_item(item);count=len(timeline['pts'])
    from production_effort import provider_first
    if provider_first(case) and any(p.get('blend_mode') != 'native_replace' for p in spec['parts']):
        raise ValueError('New orders use native interval refill, without the retired local mask pipeline')
    parts=[];artifacts={};covered=set();raw_pass=True;regions_pass=True
    context=region_context(state,item,spec['parts'],spec['frame_states'],spec.get('color_match','none')) if any(
        p.get('write_region_review') for p in spec['parts']) else None
    for part in spec['parts']:
        blend_mode=part.get('blend_mode','alpha')
        if blend_mode not in ('alpha','source_boundary_gradient','source_boundary_luma','gray_surface_plane','temporal_crossfade','source_chroma','native_replace'):raise ValueError('Unsupported provider blend mode')
        if blend_mode == 'native_replace' and (part.get('masks') or part.get('write_region_review') or spec.get('color_match', 'none') != 'none'):
            raise ValueError('Native interval refill does not use masks or local color reconstruction')
        if blend_mode=='source_chroma':require_source_chroma(item,spec['operation_id'],part)
        unit=provider.unit_for(state,item['id'],part['unit_id']);request=provider.verify_unit(case,unit)
        if unit['operation_id']!=spec['operation_id']:raise ValueError('Unit belongs to another operation')
        raw=unit.get('raw');review=unit.get('raw_review')
        if not raw or not review:raise ValueError('Raw return must be reviewed before composition')
        if review['status']!='pass':raw_pass=False
        region_artifacts = review_write_region(case,item,spec['operation_id'],unit,part,spec.get('color_match','none'),context)
        regions_pass &= review['status']=='pass' or bool(region_artifacts)
        if review['status']!='pass' and not region_artifacts and not spec.get('diagnostic'):
            raise ValueError('Failed/limited raw needs a bound write-region review or explicit diagnostic plan')
        artifacts.update(region_artifacts)
        lo,hi=request['source_frames_half_open']
        wl,wh=part.get('write_frames_half_open',[lo,hi])
        if not all(type(n)is int for n in (wl,wh)) or not lo<=wl<wh<=hi:
            raise ValueError('Writeback must stay inside source context')
        covered.update(range(wl,wh))
        for path,sha in [(raw['path'],raw['sha256']),(review['path'],review['sha256']),
                         (unit['request_path'],project.digest(project.project_path(unit['request_path']))),
                         *(([(part['masks'],part['masks_sha256'])]) if blend_mode != 'native_replace' else [])]:
            if project.digest(project.project_path(path))!=sha:raise ValueError('Raw/plan/review/mask artifact changed')
            artifacts[path]=sha
        raw_observations=project.read_json(project.project_path(review['path']))
        if blend_mode == 'native_replace' and (review['status'] != 'pass' or raw_observations.get('observed_frames') != list(range(hi-lo))):
            raise ValueError('Native refill requires complete passing raw review; diagnostic cannot bypass it')
        for e in raw_observations['evidence']:
            if project.digest(project.project_path(e['path']))!=e['sha256']:raise ValueError('Raw evidence changed')
            artifacts[e['path']]=e['sha256']
        if raw.get('receipt'):artifacts[raw['receipt']]=raw['receipt_sha256']
        if blend_mode != 'native_replace':
            with np.load(project.project_path(part['masks']),allow_pickle=False) as masks:
                w,h=item['dimensions']
                if masks['alpha'].shape!=(hi-lo,h,w) or masks['alpha'].dtype!=np.uint8:
                    raise ValueError('Source-space alpha must cover every context frame')
                from source_matte import validate_metadata
                validate_metadata(masks, item['source_sha256'], [w,h], [lo,hi])
                if masks['raw_to_source'].shape!=(hi-lo,3,3) or not np.isfinite(masks['raw_to_source']).all():
                    raise ValueError('Every frame needs a finite raw-to-source transform')
                if 'protected' in masks and (masks['protected'].shape!=masks['alpha'].shape or masks['protected'].dtype!=np.uint8):
                    raise ValueError('Invalid source-space protection')
                if any(abs(np.linalg.det(m))<1e-10 for m in masks['raw_to_source']):raise ValueError('Singular raw mapping')
        rt=source_timeline.require_cfr(source_timeline.build(project.project_path(raw['path'])))
        # No donor repetition, padding, trimming or interpolation. An explicit
        # bound audit may restore source PTS after tiny provider rate rounding.
        timing_mapping = part.get('timing_mapping')
        if timing_mapping:
            mapping=project.read_json(project.project_path(timing_mapping))
            reference_timeline = None
            if request.get('frame_clock_adapter'):
                from production_pipeline import interval_route
                route = interval_route(case,item['id'],spec['operation_id'])
                if (not route or request.get('interval_route_sha256') != workflow.hash_json(route)
                        or request['reference_fps'] != route['model_input_fps']):
                    raise ValueError('Unverified reference clock adapter')
                reference_timeline = source_timeline.build(project.project_path(request['reference_clip']))
            validate_frame_index_mapping(timeline,rt,[lo,hi],mapping,mapping_binding(item,unit,[lo,hi]),reference_timeline)
            artifacts[timing_mapping]=project.digest(project.project_path(timing_mapping))
            for e in mapping['evidence']:
                if project.digest(project.project_path(e['path']))!=e['sha256']:
                    raise ValueError('Frame correspondence evidence changed')
                artifacts[e['path']]=e['sha256']
        else:
            if len(rt['pts'])!=hi-lo or Fraction(rt['fps'])!=Fraction(timeline['fps']):
                raise ValueError('Raw duration/frame rate differs; padding/retiming forbidden')
            local_pts=[Fraction(t-timeline['pts'][lo])*Fraction(timeline['time_base']) for t in timeline['pts'][lo:hi]]
            raw_pts=[Fraction(t)*Fraction(rt['time_base']) for t in rt['pts']]
            tolerance=Fraction(timeline['time_base'])+Fraction(rt['time_base'])
            if any(abs(a-b)>tolerance for a,b in zip(local_pts,raw_pts)):
                raise ValueError('Raw PTS mapping differs')
        parts.append(dict(unit_id=unit['unit_id'],raw=raw['path'],raw_sha256=raw['sha256'],
            source_frames_half_open=[lo,hi],write_frames_half_open=[wl,wh],
            **(dict(masks=part['masks'],masks_sha256=part['masks_sha256']) if blend_mode != 'native_replace' else {}),
            provider_task_id=unit['provider_task_id'],
            request_fingerprint=unit['request_fingerprint'],raw_review=review,
            mode=raw['mode'],blend_mode=blend_mode,
            **({'target_rgb':part['target_rgb']} if blend_mode=='source_chroma' else {}),
            **({'timing_mapping':timing_mapping} if timing_mapping else {}),
            **({'write_region_review':part['write_region_review']} if region_artifacts else {})))
    require_disjoint_parts(parts)
    states=spec['frame_states']
    if len(states)!=count or any(s not in ('visible','occluded','absent','unresolved') for s in states):
        raise ValueError('Full source visibility inventory required')
    if any(s=='visible' and n not in covered for n,s in enumerate(states)):
        raise ValueError('Visible target lacks a provider unit')
    if not spec.get('observation_evidence'):raise ValueError('Visibility needs source-bound observation evidence')
    for e in spec['observation_evidence']:
        if project.digest(project.project_path(e['path']))!=e['sha256']:raise ValueError('Observation evidence changed')
        artifacts[e['path']]=e['sha256']
    for name in ('scripts/aigc_composite.py','scripts/execution_plan.py'):
        artifacts[name]=project.digest(project.project_path(name))
    plan=dict(file_id=item['id'],operation_id=spec['operation_id'],case_version=case['case_version'],
        source_sha256=item['source_sha256'],dependency=workflow.requirement_binding(case,item),
        source_frame_count=count,source_frames_half_open=[0,count],composition_mode='aigc_regional',
        mask_space='source',parts=parts,artifact_hashes=artifacts,diagnostic=bool(spec.get('diagnostic')),
        raw_reviews_pass=raw_pass,write_regions_reviewed=bool(regions_pass),color_match=spec.get('color_match','none'),review={'frame_states':states,
            'records':spec.get('frame_records') or [dict(frame=n,state=s) for n,s in enumerate(states)]},accepted=False)
    path=project.project_path(spec['out']);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(plan,stream,ensure_ascii=False,indent=2)
    return dict(plan=spec['out'],plan_sha256=project.digest(path),status='plan_pending_full_candidate_review')


def validate(case,item,plan):
    if plan.get('color_match','none') not in ('none','masked_flat_median'):
        raise ValueError('Unsupported provider color match')
    if plan.get('dependency')!=workflow.requirement_binding(case,item):raise ValueError('AIGC requirements changed')
    for p,h in plan['artifact_hashes'].items():
        if project.digest(project.project_path(p))!=h:raise ValueError('AIGC artifact changed: '+p)
    state=project.read_json(workflow.state_path(case));workflow.verify_state_case(case,state)
    raw_pass=True;regions_pass=True;covered=set()
    context=region_context(state,item,plan['parts'],plan['review']['frame_states'],plan.get('color_match','none')) if any(
        p.get('write_region_review') for p in plan['parts']) else None
    for part in plan['parts']:
        if part.get('blend_mode','alpha') not in ('alpha','source_boundary_gradient','source_boundary_luma','gray_surface_plane','temporal_crossfade','source_chroma','native_replace'):raise ValueError('Unsupported provider blend mode')
        if part.get('blend_mode')=='source_chroma':require_source_chroma(item,plan['operation_id'],part)
        unit=provider.unit_for(state,item['id'],part['unit_id']);request=provider.verify_unit(case,unit)
        if part.get('blend_mode') == 'native_replace':
            observed = project.read_json(project.project_path(unit['raw_review']['path']))
            a,b = request['source_frames_half_open']
            if unit['raw_review']['status'] != 'pass' or observed.get('observed_frames') != list(range(b-a)):
                raise ValueError('Native refill requires complete passing raw review')
            if part.get('masks') or part.get('write_region_review') or plan.get('color_match', 'none') != 'none':
                raise ValueError('Native refill has no local mask repair stage')
        if unit['raw_review']['status']!='pass':raw_pass=False
        region_artifacts=review_write_region(case,item,plan['operation_id'],unit,part,plan.get('color_match','none'),context)
        if any(plan['artifact_hashes'].get(p)!=h for p,h in region_artifacts.items()):
            raise ValueError('Write-region review artifacts changed')
        regions_pass &= unit['raw_review']['status']=='pass' or bool(region_artifacts)
        if (part['source_frames_half_open']!=request['source_frames_half_open'] or part['raw']!=unit['raw']['path']
                or (part.get('blend_mode') != 'native_replace' and plan['artifact_hashes'].get(part['masks'])!=part['masks_sha256'])):
            raise ValueError('AIGC unit mapping changed')
        if unit['operation_id']!=plan['operation_id'] or unit['raw']['sha256']!=part['raw_sha256'] or unit.get('raw_review')!=part['raw_review']:
            raise ValueError('AIGC unit/raw review changed')
        lo,hi=request['source_frames_half_open'];wl,wh=part['write_frames_half_open']
        if not all(type(n)is int for n in (wl,wh)) or not lo<=wl<wh<=hi:
            raise ValueError('AIGC write ranges overlap or exceed context')
        covered.update(range(wl,wh))
    require_disjoint_parts(plan['parts'])
    if any(s=='visible' and n not in covered for n,s in enumerate(plan['review']['frame_states'])):
        raise ValueError('Visible target lacks a provider unit')
    if raw_pass!=plan['raw_reviews_pass']:raise ValueError('AIGC raw verdict cannot be overridden')
    if plan.get('write_regions_reviewed',False) and not regions_pass:
        raise ValueError('Write-region verdict cannot be overridden')
    if not raw_pass and not (regions_pass and plan.get('write_regions_reviewed')) and not plan['diagnostic']:
        raise ValueError('Failed raw requires diagnostic or verified write regions')
    return plan['artifact_hashes']


def require_source_chroma(item,operation_id,part):
    operation=next(o for o in item['operations'] if o['id']==operation_id)
    rgb=part.get('target_rgb')
    if (operation.get('recolor_mode')!='preserve_source_details'
            or rgb!=operation.get('target_color_rgb') or not isinstance(rgb,list)
            or len(rgb)!=3 or any(type(v)is not int or not 0<=v<=255 for v in rgb)):
        raise ValueError('Source chroma must use the requested source-preserving RGB')


def source_chroma(frame,rgb):
    """Existing HSV chroma grade: retain every source value, texture and position."""
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    target=cv2.cvtColor(np.array([[rgb[::-1]]],np.uint8),cv2.COLOR_BGR2HSV)[0,0]
    hsv[:,:,0]=target[0];hsv[:,:,1]=target[1]
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


class Layer:
    def __init__(self,plan,report):
        self.plan=plan;self.report=report;self.parts=[];self.mask_cache=None
        report.update(method='aigc_regional',raw_reviews_pass=plan['raw_reviews_pass'],
                      write_regions_reviewed=plan.get('write_regions_reviewed',False),
                      diagnostic=plan['diagnostic'],provider_units=[p['unit_id'] for p in plan['parts']],
                      provider_raw_hashes={p['unit_id']:p['raw_sha256'] for p in plan['parts']})
        try:
            for p in plan['parts']:
                cap=cv2.VideoCapture(str(project.project_path(p['raw'])))
                if not cap.isOpened():raise ValueError('Cannot decode original provider raw')
                self.parts.append((p,cap))
        except Exception:
            self.close();raise

    def compose(self,frame,n):
        from regional_composite import compose_frame
        result=frame.copy();union=np.zeros(frame.shape[:2],bool)
        state=self.plan['review']['frame_states'][n]
        active=sum(p['write_frames_half_open'][0]<=n<p['write_frames_half_open'][1]
                   for p,_ in self.parts) if state=='visible' else 0
        groups={}
        for p,cap in self.parts:
            lo,hi=p['source_frames_half_open'];wl,wh=p['write_frames_half_open']
            if not wl<=n<wh or state!='visible':continue
            if p.get('blend_mode') == 'native_replace':
                k = n-lo
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != k:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
                ok, raw = cap.read()
                if not ok:
                    raise ValueError('Provider raw ended early')
                h,w = frame.shape[:2]
                if raw.shape[:2] != (h,w):
                    raw = cv2.resize(raw, (w,h), interpolation=cv2.INTER_LINEAR)
                self.report['modified_source_frames'].append(n)
                self.report['modified_instance_count'] += 1
                return raw, np.ones((h,w), bool)
            if self.mask_cache is None:self.mask_cache={}
            if p['masks'] not in self.mask_cache:
                # Two overlapping contexts stay resident; do not decompress a full
                # source-size mask archive twice per overlap frame.
                if len(self.mask_cache)>=2:del self.mask_cache[next(iter(self.mask_cache))]
                with np.load(project.project_path(p['masks']),allow_pickle=False) as z:
                    self.mask_cache[p['masks']]={k:z[k] for k in z.files}
            m=self.mask_cache[p['masks']]
            k=n-lo
            # Sequential export normally advances one frame; previews may start later.
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES))!=k:cap.set(cv2.CAP_PROP_POS_FRAMES,k)
            ok,raw=cap.read()
            if not ok:raise ValueError('Provider raw ended early')
            h,w=frame.shape[:2];matrix=m['raw_to_source'][k]
            raw=cv2.warpPerspective(raw,matrix,(w,h))
            # The native route uses a constant transform for an entire context.
            # Reuse its verified pixel support, rather than warping it every frame.
            if not hasattr(self,'support_cache'):self.support_cache={}
            key=(int(cap.get(4)),int(cap.get(3)),h,w,matrix.tobytes())
            if key not in self.support_cache:
                if len(self.support_cache)>=4:self.support_cache.clear()
                self.support_cache[key]=cv2.warpPerspective(np.full(key[:2],255,np.uint8),matrix,(w,h))
            support=self.support_cache[key]
            protected=m['protected'][k] if 'protected'in m else None
            alpha=m['alpha'][k]
            if protected is not None:
                alpha=alpha.copy();alpha[protected>0]=0
            if np.any((alpha>0)&(support<255)):raise ValueError('Writeback maps outside real provider pixels')
            mode=p.get('blend_mode','alpha')
            if mode=='source_chroma':raw=source_chroma(frame,p['target_rgb'])
            if mode in ('temporal_crossfade','source_chroma') and active==1 and np.all(alpha==255):
                # Exactly the same pixels as weighted/255 + opaque composition.
                # Partial masks, protections and context overlaps use the full path.
                self.report['modified_source_frames'].append(n)
                self.report['modified_instance_count']+=1
                return raw,np.ones(frame.shape[:2],bool)
            if not np.any(alpha):
                if mode in ('temporal_crossfade','source_chroma'):continue
                raise ValueError('Visible target has no effective provider pixels')
            if mode in ('temporal_crossfade','source_chroma'):
                if mode not in groups:groups[mode]=(np.zeros_like(frame,dtype=np.float32),np.zeros(frame.shape[:2],np.uint16))
                weighted,combined=groups[mode]
                weighted+=raw.astype(np.float32)*alpha[:,:,None]
                combined+=alpha
                if np.any(combined>255):raise ValueError('Crossfade alpha exceeds 255')
                continue
            if mode in ('source_boundary_gradient','source_boundary_luma'):
                if mode not in groups:groups[mode]=(frame.copy(),np.zeros(frame.shape[:2],np.uint8))
                donor,combined=groups[mode]
                if self.plan.get('color_match')=='masked_flat_median' and np.any(combined):
                    boundary=(cv2.dilate((alpha>0).astype(np.uint8),np.ones((7,7),np.uint8))>0)&(combined>0)&(support==255)
                    if np.count_nonzero(boundary)>=20:
                        offset=np.clip(np.median(donor[boundary].astype(np.float32)-raw[boundary],axis=0),-40,40)
                        raw=np.clip(raw.astype(np.float32)+offset,0,255).astype(np.uint8)
                touched=alpha>0
                if np.any(touched&(combined>0)):raise ValueError('Overlapping provider pixels are unresolved')
                donor[touched]=raw[touched];combined[touched]=alpha[touched]
                continue
            if mode=='gray_surface_plane':raw=match_gray_surface_plane(frame,raw,alpha,support)
            elif self.plan.get('color_match')=='masked_flat_median':raw=match_masked_flat_color(frame,raw,alpha)
            patch,effective=compose_frame(frame,raw,alpha)
            touched=effective>0
            result[touched]=patch[touched];union|=touched
        for mode,(raw,alpha) in groups.items():
            # Clone the combined donor once so an internal donor boundary cannot
            # reintroduce source branding through independent Poisson boundaries.
            if mode in ('temporal_crossfade','source_chroma'):
                raw=np.rint(raw/np.maximum(alpha[:,:,None],1)).astype(np.uint8)
                alpha=alpha.astype(np.uint8);matched=raw
            else:matched=match_source_boundary_gradient(frame,raw,alpha,mode=='source_boundary_luma')
            patch,effective=compose_frame(frame,matched,alpha)
            touched=effective>0
            result[touched]=patch[touched];union|=touched
        if np.any(union):
            self.report['modified_source_frames'].append(n);self.report['modified_instance_count']+=1
        return result,union

    def close(self):
        for _,cap in self.parts:cap.release()
        self.mask_cache=None
        self.support_cache={}


def match_masked_flat_color(source,raw,alpha):
    """Balance actual provider pixels against flat source material per mask island."""
    result=raw.copy()
    count,labels,stats,_=cv2.connectedComponentsWithStats((alpha>0).astype(np.uint8))
    gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    for index in range(1,count):
        x,y,w,h,area=stats[index]
        if area<40:continue
        island=labels[y:y+h,x:x+w]==index
        sample=island&(alpha[y:y+h,x:x+w]>=240)&(gradient[y:y+h,x:x+w]<12)
        if np.count_nonzero(sample)<30:continue
        a=source[y:y+h,x:x+w];b=raw[y:y+h,x:x+w]
        offset=np.clip(np.median(a[sample].astype(np.float32)-b[sample],axis=0),-40,40)
        box=result[y:y+h,x:x+w]
        box[island]=np.clip(b[island].astype(np.float32)+offset,0,255).astype(np.uint8)
    return result


def match_source_boundary_gradient(source,raw,alpha,monochrome=False):
    """Blend provider gradients against the original material at the mask boundary."""
    mask=np.where(alpha>0,255,0).astype(np.uint8)
    x,y,w,h=cv2.boundingRect(mask)
    if w<3 or h<3:return raw
    # Cropping is required: full-frame clone placement shifts a noncentral mask.
    left=max(0,x-2);top=max(0,y-2);right=min(source.shape[1],x+w+2);bottom=min(source.shape[0],y+h+2)
    return cv2.seamlessClone(raw[top:bottom,left:right],source,
        mask[top:bottom,left:right].copy(),(x+w//2,y+h//2),cv2.MONOCHROME_TRANSFER if monochrome else cv2.NORMAL_CLONE)


def match_gray_surface_plane(source,raw,alpha,support=None):
    """Match cached provider texture to gray-surface lighting after rejecting dark ink.

    Only an opt-in neutral material uses this source-derived illumination estimate.
    Provider pixels supply the texture; the original hand is excluded by the mask.
    """
    mask=alpha>0
    x,y,w,h=cv2.boundingRect(mask.astype(np.uint8))
    if w<3 or h<3:raise ValueError('Gray surface fit needs a visible region')
    if support is not None and not np.any((support==255)&~mask):
        raise ValueError('Insufficient neutral source boundary for color fit')
    left=max(0,x-32);top=max(0,y-32);right=min(source.shape[1],x+w+32);bottom=min(source.shape[0],y+h+32)
    src=source[top:bottom,left:right].astype(np.float32);donor=raw[top:bottom,left:right].astype(np.float32)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41))
    clean=cv2.morphologyEx(src,cv2.MORPH_CLOSE,kernel)
    illumination=cv2.GaussianBlur(clean,(21,21),5)
    donor_light=cv2.GaussianBlur(donor,(31,31),8)
    result=raw.copy();box=result[top:bottom,left:right];active=mask[top:bottom,left:right]
    box[active]=np.clip(np.rint(donor+illumination-donor_light),0,255).astype(np.uint8)[active]
    return result
