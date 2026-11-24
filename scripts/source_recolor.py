"""Recolor a measured source matte without regenerating or moving source pixels."""
import cv2
import numpy as np
import project
import workflow


def prepare_plan(case, spec):
    """Bind an imported video matte and a text/asset color to the common export."""
    from production_effort import require_lightweight_edit
    require_lightweight_edit(case)
    import json
    from video_segmentation import validate_import
    item=workflow.item_for(case,spec['file_id'])
    sidecar_path=spec['segmentation_import']
    sidecar=project.read_json(project.project_path(sidecar_path))
    from production_effort import source_preserving_recolor
    operation=next(op for op in item['operations'] if op['id']==spec['operation_id'])
    if source_preserving_recolor(operation) and (spec.get('color_rgb')!=operation['target_color_rgb']
            or sidecar['backend']!='sam3-video-semantic' or spec.get('color_asset')):
        raise ValueError('Source-preserving recolor must reuse frozen RGB and fixed SAM3 masks')
    lo,hi=sidecar['frames_half_open']
    count=item['source_frame_count_inventory']
    if (lo,hi)!=(0,count):raise ValueError('Whole-video recolor requires the complete tracked source')
    states=spec['frame_states']
    if len(states)!=count or any(s not in ('visible','partial','absent') for s in states):
        raise ValueError('Require actual whole-source visibility observations')
    for e in spec['evidence']:
        if project.digest(project.project_path(e['path']))!=e['sha256']:raise ValueError('Source observation changed')
    out=project.project_path(spec['out']);pixels=out.with_name(out.stem+'-pixels.json')
    plan=dict(file_id=item['id'],operation_id=spec['operation_id'],case_version=case['case_version'],
        source_sha256=item['source_sha256'],dependency=workflow.requirement_binding(case,item),
        composition_mode='source_recolor',source_frame_count=count,source_frames_half_open=[0,count],
        edit_frames_half_open=[lo,hi],masks=sidecar['masks'],color_asset=spec.get('color_asset'),
        color_rgb=spec.get('color_rgb'),recolor_mode='hsv_hue',
        artifact_hashes={p:project.digest(project.project_path(p)) for p in [sidecar['masks'],sidecar_path]},
        segmentation_backend=sidecar['backend'],segmentation_import=sidecar_path,
        segmentation_import_sha256=project.digest(project.project_path(sidecar_path)),full_appearance_required=True,
        review=dict(frame_states=states,records=[dict(frame=n,state=s) for n,s in enumerate(states)],
            evidence=spec['evidence'],status='source_visibility_observed'))
    if 'editable_source_hsv' in spec:
        plan['editable_source_hsv']=spec['editable_source_hsv']
    plan['segmentation_protection']=[]
    for path in spec.get('segmentation_protection_imports', []):
        guard=project.read_json(project.project_path(path))
        plan['segmentation_protection'].append(dict(masks=guard['masks'],
            edit_frames_half_open=guard['frames_half_open'],segmentation_backend=guard['backend'],
            segmentation_import=path,segmentation_import_sha256=project.digest(project.project_path(path)),
            artifact_hashes={p:project.digest(project.project_path(p)) for p in [path,guard['masks']]}))
    if not plan['segmentation_protection']:
        del plan['segmentation_protection']
    validate_import(case,item,plan);validate(case,item,plan)
    manifest=dict(schema='execution_manifest_v1',file_id=item['id'],dependency=plan['dependency'],
        operations=[dict(operation_id=spec['operation_id'],disposition='ready',plan=str(pixels.relative_to(project.ROOT)))],
        risk_preview_ranges_half_open=spec['risk_preview_ranges_half_open'])
    if out.exists() or pixels.exists():
        if not out.is_file() or not pixels.is_file() or project.read_json(pixels)!=plan:
            raise ValueError('Preserve previous plans; inputs changed or preparation incomplete')
        manifest['operations'][0]['plan_sha256']=project.digest(pixels)
        if project.read_json(out)!=manifest:
            raise ValueError('Preserve previous plans; manifest changed')
        return dict(status='prepared_pending_preview',plan=spec['out'],execution_reused=True,accepted=False,new_model_calls=0)
    from production_pipeline import guard_upstream
    guard_upstream(case,item['id'],operation_id=spec['operation_id'])
    out.parent.mkdir(parents=True,exist_ok=True);pixels.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    manifest['operations'][0]['plan_sha256']=project.digest(pixels)
    out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return dict(status='prepared_pending_preview',plan=spec['out'],accepted=False,new_model_calls=0)


def protection_mask(mask, envelope=None):
    """Close semantic-object holes, never infer a target from global colour."""
    if envelope is None:
        return mask
    if envelope not in ('convex_object', 'convex_components'):
        raise ValueError('Unknown semantic protection envelope')
    output = np.zeros_like(mask)
    if envelope == 'convex_object':
        points = cv2.findNonZero(mask)
        if points is not None:
            cv2.fillConvexPoly(output, cv2.convexHull(points), 255)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.fillConvexPoly(output, cv2.convexHull(contour), 255)
    return output


def material_protection(frame, mask, material):
    if material is None:
        return mask
    if material != 'white_ink':
        raise ValueError('Unknown protected material')
    # The carrier is already localized/tracked. Separate its contrasting print,
    # rather than preserve a rectangular patch of the old fabric colour.
    selected = mask != 0
    values = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[selected]
    if not len(values):
        return mask
    threshold, _ = cv2.threshold(values,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (selected & (cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) > threshold) &
        (hsv[:,:,1] < 80)).astype(np.uint8)*255


def green_material_mask(frame):
    """Keep trim highlights and subsampled chroma edges near a green seed."""
    hue,sat,_ = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    green = (hue >= 35) & (hue <= 85)
    seed = (green & (sat >= 110)).astype(np.uint8)*255
    # Lowering saturation globally also keys neutral fabric. Limit the weaker
    # colour to the two source pixels around a confident, contrasting trim.
    near = cv2.dilate(seed, np.ones((5,5), np.uint8)) != 0
    return (green & ((sat >= 110) | (near & (sat >= 60)))).astype(np.uint8)*255


def validate(case, item, plan):
    if plan.get('native_colour_guide'):
        from native_colour_guide import validate as validate_native_guide
        validate_native_guide(case,item,plan)
    from production_effort import check, source_preserving_recolor
    check(case, item['id'], method_family='local_recolor')
    dependencies = dict(plan['artifact_hashes'])
    if plan.get('alpha_refinement'):
        from alpha_refinement import validate as validate_alpha
        dependencies.update(validate_alpha(case, item, plan))
    if plan.get('dependency') != workflow.requirement_binding(case, item):
        raise ValueError('Source recolor requirements changed')
    operation = next(o for o in item['operations'] if o['id'] == plan['operation_id'])
    preserving = source_preserving_recolor(operation)
    local_repair = None
    if plan.get('segmentation_backend') == 'cached_local_material_repair':
        from local_material_repair import verify, verify_plan
        entry = project.read_json(workflow.state_path(case))['files'][item['id']]
        local_repair = verify(case, item['id'], entry)
        if not local_repair:
            raise ValueError('Local colour repair requires preserved diagnosed parents')
        verify_plan(case, item['id'], entry, plan)
    gate=plan.get('editable_source_hsv')
    if gate is not None and (not isinstance(gate,dict) or set(gate)!={'hue_range','minimum_saturation'} or
            not isinstance(gate['hue_range'],list) or len(gate['hue_range'])!=2 or
            any(type(v) is not int for v in gate['hue_range']+[gate['minimum_saturation']]) or
            not 0 <= gate['hue_range'][0] < gate['hue_range'][1] <= 179 or
            not 0 <= gate['minimum_saturation'] <= 255 or plan.get('segmentation_backend')!='sam3-video-semantic'):
        raise ValueError('Material colour gate requires a bound SAM3 object and explicit HSV range')
    if preserving:
        if (plan.get('color_rgb') != operation['target_color_rgb'] or plan.get('color_asset') or
                (not local_repair and (plan.get('recolor_mode') != 'hsv_hue' or plan.get('segmentation_backend') != 'sam3-video-semantic')) or
                (local_repair and plan.get('recolor_mode') not in ('hsv_hue', 'hsv_chroma', 'luma_chroma')) or
                plan.get('full_appearance_required') is not True or
                plan.get('edit_frames_half_open') != [0,item['source_frame_count_inventory']]):
            raise ValueError('Source-preserving recolor must reuse frozen RGB, hue mode, fixed SAM3 and full coverage')
        evidence = plan.get('review',{}).get('evidence',[])
        if not evidence:
            raise ValueError('Source-preserving recolor requires source observation evidence')
        for e in evidence:
            if project.digest(project.project_path(e['path'])) != e['sha256']:
                raise ValueError('Source observation changed')
            dependencies[e['path']]=e['sha256']
    if plan.get('color_asset') != operation.get('reference_asset'):
        raise ValueError('Recolor reference does not belong to operation')
    if not plan.get('color_asset') and (not isinstance(plan.get('color_rgb'),list) or
            len(plan['color_rgb']) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in plan['color_rgb'])):
        raise ValueError('Text-only color changes require an explicit RGB target')
    for path, expected in plan['artifact_hashes'].items():
        if project.digest(project.project_path(path)) != expected:
            raise ValueError('Source recolor artifact changed: ' + path)
    if not local_repair and (preserving or case.get('execution_policy', {}).get('version') == 2):
        from video_segmentation import validate_import
        validate_import(case, item, plan)
        if plan.get('alpha_envelope'):
            envelope=plan['alpha_envelope']
            if envelope.get('foreground_occluders'):
                state=project.read_json(workflow.state_path(case))
                from runninghub_adapter import unit_for,verify_unit
                sidecar=project.read_json(project.project_path(plan['segmentation_import']))
                request=verify_unit(case,unit_for(state,item['id'],sidecar['unit_id']))
                if request.get('model_id')!='vitmatte-video-local':
                    raise ValueError('Occluder unknown band requires a learned matting request')
                p=request['parameters']
                if (envelope.get('unknown_band')!=p['edge_dilate'] or
                        envelope.get('unknown_band_grid')!=[p['width'],p['height']]):
                    raise ValueError('Occluder context must use the bound matting trimap grid/band')
            for region in [envelope]+envelope.get('foreground_occluders',[]):
                validate_import(case,item,dict(region,operation_id=plan['operation_id']))
                if region is envelope and (region['edit_frames_half_open'][0]>plan['edit_frames_half_open'][0] or region['edit_frames_half_open'][1]<plan['edit_frames_half_open'][1]):
                    raise ValueError('Semantic envelope must cover the refined alpha')
                for path,expected in region['artifact_hashes'].items():
                    if project.digest(project.project_path(path))!=expected:
                        raise ValueError('Semantic alpha envelope changed')
                    dependencies[path]=expected
        for protection in plan.get('segmentation_protection', []):
            if protection.get('segmentation_backend') == 'carrier_context_lk_ecc':
                from tracked_protection import validate as validate_track
                validate_track(case, item, dict(protection, operation_id=plan['operation_id']))
            else:
                validate_import(case, item, dict(protection, operation_id=plan['operation_id']))
    for protection in plan.get('segmentation_protection', []):
        if protection.get('matting_boundary', 'fixed') not in ('fixed', 'soft'):
            raise ValueError('Unknown semantic protection matting boundary')
        if protection.get('material') not in (None, 'white_ink'):
            raise ValueError('Unknown protected material')
        protection_mask(np.zeros((2, 2), np.uint8), protection.get('envelope'))
        lo, hi = protection['edit_frames_half_open']
        if not plan['edit_frames_half_open'][0] <= lo < hi <= plan['edit_frames_half_open'][1]:
            raise ValueError('Protection mask falls outside the edit range')
        for path, expected in protection['artifact_hashes'].items():
            if project.digest(project.project_path(path)) != expected:
                raise ValueError('Protection mask dependency changed')
            if path in dependencies and dependencies[path] != expected:
                raise ValueError('Conflicting protection dependency')
            dependencies[path] = expected
    with np.load(project.project_path(plan['masks']), allow_pickle=False) as z:
        lo, hi = plan['edit_frames_half_open']
        if (not 0 <= lo < hi <= plan['source_frame_count'] or
                z['alpha'].dtype != np.uint8 or z['alpha'].ndim != 3 or len(z['alpha']) != hi-lo):
            raise ValueError('Invalid source recolor matte or range')
        if 'protected' in z and (z['protected'].shape != z['alpha'].shape or z['protected'].dtype != np.uint8):
            raise ValueError('Invalid source recolor protection matte')
        if not {'source_sha256', 'source_dimensions', 'mask_dimensions', 'frames_half_open', 'protected'} <= set(z.files):
            raise ValueError('New source recolor execution requires source/grid/range/protection metadata; prepare a bound matte')
        if plan.get('skin_guard_red_minus_blue', 256) < 256:
            raise ValueError('Use explicit protection, not a global skin colour threshold, for a bound source matte')
        from source_matte import validate_metadata
        validate_metadata(z, item['source_sha256'], item['dimensions'], [lo, hi])
        if plan.get('full_appearance_required'):
            from regional_composite import validate_full_coverage
            nonempty=[False]*plan['source_frame_count']
            effective = z['alpha'].copy()
            if 'protected' in z:
                effective[z['protected'] != 0] = 0
            for protection in plan.get('segmentation_protection', []):
                guard_lo, guard_hi = protection['edit_frames_half_open']
                with np.load(project.project_path(protection['masks']), allow_pickle=False) as guard:
                    if guard['alpha'].dtype != np.uint8 or guard['alpha'].ndim != 3 or len(guard['alpha']) != guard_hi-guard_lo:
                        raise ValueError('Invalid semantic protection matte')
                    for offset, mask in enumerate(guard['alpha']):
                        resized = cv2.resize(protection_mask(mask, protection.get('envelope')), effective.shape[2:0:-1], interpolation=cv2.INTER_NEAREST)
                        effective[guard_lo-lo+offset][resized != 0] = 0
            nonempty[lo:hi]=[bool(np.any(a)) for a in effective]
            validate_full_coverage(plan['source_frames_half_open'],plan['source_frame_count'],
                                   plan['review']['frame_states'],nonempty)
    return dependencies


class Layer:
    def __init__(self, plan, report):
        self.plan, self.report = plan, report
        self.native_guide = None
        if plan.get('native_colour_guide'):
            with np.load(project.project_path(plan['native_colour_guide']['masks']),allow_pickle=False) as z:
                self.native_guide = z['alpha']
        with np.load(project.project_path(plan['masks']), allow_pickle=False) as z:
            self.alpha = z['alpha']
            self.protected = z['protected'] if 'protected' in z else None
            self.source_size = list(z['source_dimensions']) if 'source_dimensions' in z else None
        if plan.get('color_asset'):
            asset = cv2.imread(str(project.project_path(plan['color_asset'])))
            if asset is None:
                raise ValueError('Cannot decode recolor reference')
            self.color = np.median(asset.reshape(-1, 3), axis=0).astype(np.float32)
        else:
            self.color = np.array(plan['color_rgb'][::-1],dtype=np.float32)
        self.protection = []
        for protection in plan.get('segmentation_protection', []):
            with np.load(project.project_path(protection['masks']), allow_pickle=False) as z:
                self.protection.append((protection['edit_frames_half_open'], z['alpha'],
                    protection.get('envelope'), protection.get('matting_boundary', 'fixed'), protection.get('material')))
        self.refined = None
        self.envelope = None
        self.occluder_envelopes = []
        if plan.get('alpha_envelope'):
            with np.load(project.project_path(plan['alpha_envelope']['masks']),allow_pickle=False) as z:
                self.envelope=z['alpha']
            for region in plan['alpha_envelope'].get('foreground_occluders',[]):
                with np.load(project.project_path(region['masks']),allow_pickle=False) as z:
                    self.occluder_envelopes.append((region['edit_frames_half_open'],z['alpha']))
        if plan.get('alpha_refinement'):
            with np.load(project.project_path(plan['alpha_refinement']['masks']), allow_pickle=False) as z:
                self.refined = z['alpha']
        report.update(method='source_recolor', edit_frames_half_open=plan['edit_frames_half_open'],
                      regenerated_source_geometry=False)

    def fixed_protection(self, frame, n):
        """Interior labels/trim stay immutable when a soft object edge is matted."""
        h, w = frame.shape[:2]
        lo, hi = self.plan['edit_frames_half_open']
        fixed = np.zeros((h, w), np.uint8)
        if self.native_guide is not None:
            a,b=self.plan['native_colour_guide']['source_frames_half_open']
            if a <= n < b:
                fixed[self.native_guide[n-a] == 0] = 255
        if not lo <= n < hi:
            return fixed
        if self.protected is not None:
            fixed |= cv2.resize(self.protected[n-lo], (w,h), interpolation=cv2.INTER_NEAREST)
        for (guard_lo, guard_hi), masks, envelope, boundary, material in self.protection:
            if boundary == 'fixed' and guard_lo <= n < guard_hi:
                guard=cv2.resize(protection_mask(masks[n-guard_lo], envelope), (w,h), interpolation=cv2.INTER_NEAREST)
                fixed |= material_protection(frame,guard,material)
        if self.plan.get('preserve_green_and_white'):
            fixed |= green_material_mask(frame)
        return fixed

    def mask(self, frame, n):
        alpha=self.object_mask(frame,n)
        if self.native_guide is not None:
            a,b=self.plan['native_colour_guide']['source_frames_half_open']
            alpha = np.minimum(alpha,self.native_guide[n-a]) if a <= n < b else np.zeros_like(alpha)
        gate=self.plan.get('editable_source_hsv')
        if gate:
            # Restrict material inside SAM3; source colour never locates an object.
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            alpha[(hsv[:,:,0]<gate['hue_range'][0]) | (hsv[:,:,0]>gate['hue_range'][1]) |
                  (hsv[:,:,1]<gate['minimum_saturation'])]=0
        return alpha

    def object_mask(self, frame, n):
        lo, hi = self.plan['edit_frames_half_open']
        if not lo <= n < hi:
            return np.zeros(frame.shape[:2], np.uint8)
        h, w = frame.shape[:2]
        if self.source_size is not None and self.source_size != [w, h]:
            raise ValueError('Source matte dimensions differ from decoded frame')
        if self.refined is not None or self.plan.get('segmentation_backend') == 'vitmatte-video-local':
            refined = self.refined if self.refined is not None else self.alpha
            alpha = cv2.resize(refined[n-lo], (w, h), interpolation=cv2.INTER_LINEAR)
            if self.envelope is not None:
                offset=n-self.plan['alpha_envelope']['edit_frames_half_open'][0]
                envelope=cv2.resize(self.envelope[offset],(w,h),interpolation=cv2.INTER_LINEAR)
                # Foreground cutouts are resolved by learned alpha, not by the
                # coarser semantic boundary. Exterior background remains capped.
                for (start,end),masks in self.occluder_envelopes:
                    if start <= n < end:
                        guard=masks[n-start]
                        band=self.plan['alpha_envelope'].get('unknown_band',0)
                        if band:
                            guard=cv2.resize(guard,tuple(self.plan['alpha_envelope']['unknown_band_grid']),interpolation=cv2.INTER_NEAREST)
                            guard=cv2.dilate(guard,np.ones((2*band+1,2*band+1),np.uint8))
                        envelope=np.maximum(envelope,cv2.resize(guard,(w,h),interpolation=cv2.INTER_LINEAR))
                alpha=np.minimum(alpha,envelope)
            alpha[self.fixed_protection(frame, n) != 0] = 0
            # A learned edge cannot override a confidently segmented hand.
            # Leave one mask pixel uncertain instead of hardening its contour.
            for (start, end), masks, envelope, boundary, material in self.protection:
                if boundary == 'soft' and start <= n < end:
                    core = cv2.erode(protection_mask(masks[n-start], envelope), np.ones((3,3), np.uint8))
                    core = cv2.resize(core, (w,h), interpolation=cv2.INTER_NEAREST)
                    alpha[material_protection(frame, core, material) == 255] = 0
            return alpha
        alpha = cv2.resize(self.alpha[n-lo], (w, h), interpolation=cv2.INTER_LINEAR)
        # Retain the old white-shirt heuristic only for archived legacy plans.
        # New execution validates a measured protection matte instead.
        delta = frame[:, :, 2].astype(np.int16)-frame[:, :, 0].astype(np.int16)
        if 'skin_guard_red_minus_blue' in self.plan:
            alpha[delta > self.plan['skin_guard_red_minus_blue']] = 0
        if self.protected is not None:
            guard = cv2.resize(self.protected[n-lo], (w, h), interpolation=cv2.INTER_NEAREST)
            alpha[guard != 0] = 0
        for (guard_lo, guard_hi), masks, envelope, boundary, material in self.protection:
            if guard_lo <= n < guard_hi:
                guard = cv2.resize(protection_mask(masks[n-guard_lo], envelope), (w,h), interpolation=cv2.INTER_NEAREST)
                alpha[material_protection(frame,guard,material) != 0] = 0
        if self.plan.get('preserve_green_and_white'):
            # Material guards operate only inside the tracked object matte.
            # They preserve contrasting trims; they never locate the backpack.
            alpha[green_material_mask(frame) != 0] = 0
        return alpha

    def compose(self, frame, n):
        alpha = self.mask(frame, n)
        weight = alpha.astype(np.float32)[:, :, None]/255
        if self.plan.get('recolor_mode') == 'hsv_hue':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            target = cv2.cvtColor(self.color.astype(np.uint8).reshape(1,1,3), cv2.COLOR_BGR2HSV)[0,0]
            hsv[:,:,0] = target[0]
            recolored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif self.plan.get('recolor_mode') == 'hsv_chroma':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            target = cv2.cvtColor(self.color.astype(np.uint8).reshape(1,1,3), cv2.COLOR_BGR2HSV)[0,0]
            hsv[:,:,0] = target[0]
            hsv[:,:,1] = target[1]
            recolored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif self.plan.get('recolor_mode') == 'luma_chroma':
            luma = np.dot(frame.astype(np.float32), [.114, .587, .299])[:, :, None]
            target_luma = float(np.dot(self.color, [.114, .587, .299]))
            recolored = np.clip(luma*self.color/max(target_luma, 1), 0, 255)
        elif self.plan.get('recolor_mode') == 'hsv_reference':
            # Change fabric chroma while retaining source shading and pixel positions.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            target = cv2.cvtColor(self.color.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            hsv[:, :, 0] = target[0]
            hsv[:, :, 1] *= float(target[1])/self.plan['reference_saturation']
            hsv[:, :, 2] *= float(target[2])/self.plan['reference_value']
            recolored = cv2.cvtColor(np.rint(np.clip(hsv, 0, 255)).astype(np.uint8), cv2.COLOR_HSV2BGR)
        else:
            # Channelwise multiplication preserves source folds/texture and coordinates.
            recolored = np.clip(frame.astype(np.float32)*self.color/self.plan['reference_white'], 0, 255)
        output = np.rint(frame*(1-weight)+recolored*weight).astype(np.uint8)
        touched = alpha > 0
        if np.any(touched):
            self.report['modified_source_frames'].append(n)
            self.report['modified_instance_count'] += 1
        return output, touched
