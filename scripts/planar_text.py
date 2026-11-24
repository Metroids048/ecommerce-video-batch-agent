"""Source-bound planar text replacement. The host Agent supplies a visual seed.

SIFT scans every decoded frame, including cuts and mirrored reappearances. A
non-match is an unresolved observation, never proof that a requested target is
absent. Compact masks are in the label plane, not full-frame RAM arrays.
"""
import argparse
import json
import time

import cv2
import numpy as np

import project


def distinct_instances(instances):
    found=[]
    for instance in sorted(instances,key=lambda i:i['inliers'],reverse=True):
        q=np.float32(instance['source_quad'])
        duplicate=False
        for prior in found:
            p=np.float32(prior['source_quad'])
            overlap,_=cv2.intersectConvexConvex(cv2.convexHull(q),cv2.convexHull(p))
            length=max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[1]))
            if (overlap/max(1,min(abs(cv2.contourArea(q)),abs(cv2.contourArea(p))))>.2 and
                    np.linalg.norm(q.mean(axis=0)-p.mean(axis=0))<length*.15):
                duplicate=True;break
        if not duplicate:found.append(instance)
    return found


def carrier_instances(refkeys, refdesc, quad, keys, descriptors, carrier_area,
                      width, height, minimum_spread=.30, scale=1):
    """Match every destination to the reference, then separate geometric models.

    Reference-to-destination ratio tests discarded repeated identical carriers:
    the second real instance became the nearest-match competitor. Reversing
    the query retains both instances; each still needs independent carrier
    support outside the old word.
    """
    pairs=cv2.BFMatcher().knnMatch(descriptors,refdesc,k=2)
    good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.70*b.distance]
    src=np.float32([refkeys[m.trainIdx].pt for m in good])
    dst=np.float32([keys[m.queryIdx].pt for m in good])
    found=[]
    while len(src)>=12:
        matrix,inliers=cv2.findHomography(src,dst,cv2.RANSAC,2.5)
        if matrix is None or inliers is None:break
        selected=inliers.ravel()!=0
        support=int(selected.sum())
        if support<12:break
        supported=src[selected]
        spread=abs(cv2.contourArea(cv2.convexHull(supported)))/carrier_area
        projected=cv2.perspectiveTransform(np.float32(quad)[None],matrix)[0]/scale
        if (np.isfinite(projected).all() and cv2.isContourConvex(projected) and
                cv2.contourArea(projected,oriented=True)*cv2.contourArea(np.float32(quad),oriented=True)>0 and
                80<abs(cv2.contourArea(projected))<width*height*.25 and spread>=minimum_spread):
            found.append({'source_quad':projected.tolist(),'inliers':support,
                          'matches':len(src),'carrier_support_fraction':round(spread,4)})
        src,dst=src[~selected],dst[~selected]
    return distinct_instances(found)


def refine(case, parent, spec, out):
    """Fill bounded, Agent-assessed shot gaps with bidirectional optical flow.

    A lost track stays unresolved. Shot bounds and foreground polygons are
    supplied by the host Agent from the actual source, never by the customer.
    """
    if spec.get('method')=='surface_reprepare':
        from text_localization import reprepare
        return reprepare(case,parent,spec,out)
    item = next(f for f in case['files'] if f['id'] == parent['file_id'])
    if any('instances' in r for r in parent['review']['records']):
        raise ValueError('Single-carrier refinement cannot overwrite instance plans; relocalize the operation')
    if spec['source_sha256'] != item['source_sha256']:
        raise ValueError('Refinement belongs to another source')
    output = project.project_path(out)
    if output.exists():
        raise ValueError('Preserve the existing refinement version')
    with np.load(project.project_path(parent['masks']), allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files}
    cap = cv2.VideoCapture(str(project.project_path(item['source'])))
    scale = min(1, 720 / cap.get(3))
    grays = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grays.append(cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY))
    cap.release()
    records = parent['review']['records']
    for reset in spec.get('reset_ranges', []):
        lo, hi = reset['frames_half_open']
        if not 0 <= lo < hi <= len(records) or not reset.get('reason'):
            raise ValueError('Invalid rejected tracking range')
        for n in range(lo, hi):
            records[n] = {'frame': n, 'state': 'unresolved', 'agent_observation': reset['reason']}
            arrays['alpha'][n] = 0
    for segment in spec.get('tracked_segments', []):
        lo, hi = segment['frames_half_open']
        seed = segment['seed_frame']
        if not 0 <= lo <= seed < hi <= len(grays):
            raise ValueError('Invalid bounded shot')
        # A new tracking request supersedes an earlier absence assessment.
        # Lost tracks must remain visible as work, not inherit an old pass.
        seed_quad = segment.get('label_quad', records[seed].get('source_quad'))
        for n in range(lo, hi):
            record = records[n]
            if 'source_quad' not in record or segment.get('replace_existing'):
                record.update(state='unresolved')
                record.pop('agent_observation', None)
                record.pop('source_quad', None)
                arrays['alpha'][n] = 0
        quad = np.float32(seed_quad)*scale
        poly = np.float32(segment['carrier_polygon'])*scale
        mask = np.zeros(grays[seed].shape, np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        points = cv2.goodFeaturesToTrack(grays[seed], 250, .003, 4, mask=mask)
        max_distance = segment.get('max_seed_distance', 30)
        if not isinstance(max_distance, int) or not 0 <= max_distance <= 30:
            raise ValueError('Re-anchor tracking within 30 source frames')
        for direction in (-1, 1):
            for n in range(seed, lo-1 if direction < 0 else hi, direction):
                if abs(n-seed) > max_distance:
                    break
                projected = quad.copy()
                if n != seed:
                    if points is None or len(points) < 8:
                        break
                    # Estimate each frame against the observed source anchor.
                    # Accumulated per-frame affines drifted from a moving bag
                    # label onto the bag body and subtitles in a real batch.
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(grays[seed], grays[n], points, None, winSize=(25, 25), maxLevel=3)
                    back, status_back, _ = cv2.calcOpticalFlowPyrLK(grays[n], grays[seed], moved, None, winSize=(25, 25), maxLevel=3)
                    good = (status.ravel() != 0) & (status_back.ravel() != 0) & (np.linalg.norm(points-back, axis=2).ravel() < 1.5)
                    if good.sum() < 8:
                        break
                    matrix, support = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC, ransacReprojThreshold=1.5)
                    if matrix is None or int(support.sum()) < 8:
                        break
                    affine = np.vstack([matrix, [0, 0, 1]])
                    projected = cv2.perspectiveTransform(projected[None], affine)[0]
                    height, width = grays[n].shape
                    if (not np.isfinite(projected).all() or np.any(projected < 0)
                            or np.any(projected[:, 0] >= width) or np.any(projected[:, 1] >= height)):
                        break
                if 'source_quad' not in records[n] or segment.get('replace_existing'):
                    records[n].update(source_quad=(projected/scale).tolist(), state='visible',
                                      track_method='anchor_forward_backward_LK', agent_seed_frame=seed)
                    records[n].pop('agent_observation', None)
                    if 'label_style' in segment:
                        records[n]['label_style'] = segment['label_style']
                    arrays['alpha'][n] = 255
    for assessment in spec.get('assessments', []):
        lo, hi = assessment['frames_half_open']
        for n in range(lo, hi):
            if assessment['state'] in ('absent', 'occluded'):
                records[n].update(state=assessment['state'], agent_observation=assessment['reason'])
                records[n].pop('source_quad', None)
                arrays['alpha'][n] = 0
    for observation in spec.get('observed_frames', []):
        n = observation['frame']
        quad = np.float32(observation['source_quad'])
        if not 0 <= n < len(records) or quad.shape != (4, 2) or not np.isfinite(quad).all():
            raise ValueError('Invalid Agent frame observation')
        records[n].update(source_quad=quad.tolist(), state='visible',
                          track_method='agent_source_frame_observation',
                          agent_observation=observation['reason'])
        arrays['alpha'][n] = 255
        if 'label_style' in observation:
            records[n]['label_style'] = observation['label_style']
    for adjustment in spec.get('style_ranges', []):
        lo, hi = adjustment['frames_half_open']
        for n in range(lo, hi):
            records[n].setdefault('label_style', {}).update(adjustment['label_style'])
    for interval in spec.get('reset_protection', []):
        arrays['protected'][interval[0]:interval[1]] = 0
    for protection in spec.get('protection', []):
        lo, hi = protection['frames_half_open']
        polygon = np.rint(protection['plane_polygon']).astype(np.int32)
        for n in range(lo, hi):
            cv2.fillPoly(arrays['protected'][n], [polygon], 255)
    for protection in spec.get('dark_foreground_protection', []):
        cap = cv2.VideoCapture(str(project.project_path(item['source'])))
        lo, hi = protection['frames_half_open']
        pw, ph = parent['label_style']['plane_size']
        plane = np.float32([[0, 0], [pw-1, 0], [pw-1, ph-1], [0, ph-1]])
        cap.set(1, lo)
        for n in range(lo, hi):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Foreground source decode failed')
            if 'source_quad' not in records[n]:
                continue
            patch = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(np.float32(records[n]['source_quad']), plane), (pw, ph))
            mask = np.uint8(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) < protection['max_gray'])*255
            if protection.get('connected_edge') == 'left':
                _, labels = cv2.connectedComponents(mask)
                foreground = np.unique(labels[:, 0])
                foreground = foreground[foreground != 0]
                mask = np.uint8(np.isin(labels, foreground))*255
            arrays['protected'][n] |= cv2.dilate(mask, np.ones((7, 7), np.uint8))
        cap.release()
    masks = output.with_suffix('.npz')
    np.savez_compressed(masks, **arrays)
    parent.update(masks=str(masks.relative_to(project.ROOT)), masks_sha256=project.digest(masks),
                  refinement=spec)
    parent['review']['frame_states'] = [r['state'] for r in records]
    parent['review']['method'] += '; Agent bounded shot refinement and label-plane foreground protection'
    output.write_text(json.dumps(parent, ensure_ascii=False, indent=2)+'\n')
    return {'plan': out, 'edited_frames': int(np.any(arrays['alpha'], axis=(1, 2)).sum()),
            'unresolved_frames': sum(r['state'] == 'unresolved' for r in records)}


def localize(case, spec, out):
    if spec.get("method") == "on_device_text":
        from text_localization import localize as localize_text
        return localize_text(case, spec, out)
    project.validate_case(case)
    item = next(f for f in case['files'] if f['id'] == spec['file_id'])
    if spec['source_sha256'] != item['source_sha256']:
        raise ValueError('Visual seed belongs to another source')
    output = project.project_path(out)
    if output.exists():
        raise ValueError('Preserve the existing localization version')
    cap = cv2.VideoCapture(str(project.project_path(item['source'])))
    count, width, height = int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    scale = min(1, 540 / width)
    size = (round(width * scale), round(height * scale))
    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.025)
    templates = []
    seeds = spec.get('seeds', [spec])
    if not seeds:
        raise ValueError('At least one visual seed required')
    for visual_seed in seeds:
        cap.set(1, visual_seed['seed_frame'])
        ok, seed = cap.read()
        if not ok:
            raise ValueError('Cannot decode the Agent seed')
        seed = cv2.resize(seed, size)
        label = np.asarray(visual_seed['label_quad'], np.float32) * scale
        polygon = np.asarray(visual_seed['carrier_polygon'], np.float32) * scale
        if (label.shape != (4, 2) or polygon.ndim != 2 or polygon.shape[1] != 2
                or len(polygon) < 3 or not np.isfinite(label).all() or not np.isfinite(polygon).all()):
            raise ValueError('Invalid visual seed geometry')
        for mirrored in (False, True) if visual_seed.get('allow_mirror', spec.get('allow_mirror')) else (False,):
            image = cv2.flip(seed, 1) if mirrored else seed
            poly, quad = polygon.copy(), label.copy()
            if mirrored:
                poly[:, 0] = size[0] - 1 - poly[:, 0]
                quad[:, 0] = size[0] - 1 - quad[:, 0]
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            cv2.fillPoly(mask, [quad.astype(np.int32)], 0)
            key, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
            if desc is None or len(desc) < 20:
                raise ValueError('Insufficient carrier texture; use another method')
            templates.append((key, desc, quad, mirrored, abs(cv2.contourArea(poly)), visual_seed['seed_frame']))
    plane_w, plane_h = spec.get('plane_size', [512, 96])
    alpha = np.zeros((count, plane_h, plane_w), np.uint8)
    protected = np.zeros_like(alpha)
    records, frame_states = [], []
    cap.set(1, 0)
    started = time.monotonic()
    for n in range(count):
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Source decode ended early')
        gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
        keys, descriptors = sift.detectAndCompute(gray, None)
        instances = []
        if descriptors is not None:
            for refkeys, refdesc, quad, mirrored, carrier_area, seed_frame in templates:
                found=carrier_instances(refkeys,refdesc,quad,keys,descriptors,carrier_area,
                                        width,height,spec.get('min_carrier_support',.30),scale)
                for instance in found:
                    instance.update(mirrored=mirrored,seed_frame=seed_frame,state='visible')
                    instances.append(instance)
        instances=distinct_instances(instances)
        if instances:
            alpha[n] = 255
            state = 'visible'
        else:
            state = 'unresolved'
        for j,instance in enumerate(instances):instance['instance_id']=f'sift-{n}-{j}'
        primary=max(instances,key=lambda i:i['inliers']) if instances else {}
        records.append(dict(primary,frame=n,state=state,instances=instances))
        frame_states.append(state)
        if n % 150 == 0:
            print(json.dumps({'decoded': n, 'total': count, 'seconds': round(time.monotonic()-started, 1)}), flush=True)
    cap.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    masks = output.with_suffix('.npz')
    np.savez_compressed(masks, alpha=alpha, protected=protected,
                        raw_to_source=np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)))
    asset = next(op['replacement_asset'] for op in item['operations'] if op['id'] == spec['operation_id'])
    plan = {'case_version': case['case_version'], 'file_id': item['id'], 'operation_id': spec['operation_id'],
            'source_sha256': item['source_sha256'], 'source_frame_count': count,
            'source_frames_half_open': [0, count], 'cloud': item['source'], 'cloud_sha256': item['source_sha256'],
            'masks': str(masks.relative_to(project.ROOT)), 'masks_sha256': project.digest(masks),
            'mask_space': 'label_plane', 'composition_mode': 'exact_asset_on_source_label',
            'exact_asset': asset, 'exact_asset_sha256': project.digest(project.project_path(asset)),
            'localization_limit': 'All geometrically supported carrier instances; nonmatches remain unresolved. Multiple instances require the unified execution manifest.',
            'label_style': {'plane_size': [plane_w, plane_h], 'polarity': 'dark', 'asset_fit': [.88, .64]},
            'parent_version': None, 'provider_task_id': None, 'full_scope_requested': False,
            'encoding_crf': 16, 'spec': spec,
            'review': {'method': 'Agent visual seed; SIFT/RANSAC per decoded frame; mirror template; unresolved nonmatches require Agent assessment',
                       'frame_states': frame_states, 'records': records, 'full_mask_reviewed': False}}
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    return {'plan': out, 'frames': count, 'localized': sum(r['state'] == 'visible' for r in records),
            'status': 'pending_agent_coverage_and_visual_review'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(localize(project.read_json(project.project_path(args.case)),
                              project.read_json(project.project_path(args.spec)), args.out), ensure_ascii=False))
