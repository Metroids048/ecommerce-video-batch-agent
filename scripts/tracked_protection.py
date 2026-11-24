"""Propagate small rigid protected labels from recorded SAM2 roots with LK/ECC.

The deformable target still uses the provider's video segmentation. This local
registration path protects a small label that the segmenter loses after reentry.
"""
import json
import cv2
import numpy as np
import project
import workflow
import carrier_tracking
import video_segmentation


def prepare(case, spec, out):
    item = workflow.item_for(case, spec['file_id'])
    root = dict(spec['root'], operation_id=spec['operation_id'])
    video_segmentation.validate_import(case, item, root)
    if project.digest(project.project_path(item['source'])) != item['source_sha256']:
        raise ValueError('Protection source changed')
    seeds = spec['seed_frames']
    if not 1 <= len(seeds) <= 8 or seeds != sorted(set(seeds)):
        raise ValueError('Use sparse observed label roots, never per-frame corrections')
    path = project.project_path(out)
    if path.exists() or path.with_suffix('.npz.json').exists() or path.suffix != '.npz':
        raise ValueError('Preserve prior protection artifacts')
    with np.load(project.project_path(root['masks']), allow_pickle=False) as z:
        masks = z['alpha']
        height, width = masks.shape[1:]
    lo, hi = root['edit_frames_half_open']
    if any(type(n) is not int or not lo <= n < hi or not masks[n-lo].any() for n in seeds):
        raise ValueError('Each label root must be a visible recorded provider mask')
    cap = cv2.VideoCapture(str(project.project_path(item['source'])))
    grays = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grays.append(cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2GRAY))
    cap.release()
    count = item['source_frame_count_inventory']
    if len(grays) != count:
        raise ValueError('Protection tracking source frame count changed')
    result = np.zeros((count, height, width), np.uint8)
    records = {}
    for seed in seeds:
        contours, _ = cv2.findContours(masks[seed-lo], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = np.concatenate(contours)
        quad = cv2.boxPoints(cv2.minAreaRect(points))
        if abs(cv2.contourArea(quad)) > width*height*.02:
            raise ValueError('Rigid label protection cannot cover the whole deformable target')
        records[seed] = dict(frame=seed, root=seed, quad=quad.tolist(), score=1.)
        for direction in (-1, 1):
            previous, q = seed, quad.copy()
            for n in range(seed+direction, count if direction == 1 else -1, direction):
                found = carrier_tracking.register(grays[previous], grays[n], q)
                if found is None:
                    break  # No interpolated guesses across loss or offscreen intervals.
                q = found['word_quad']
                # register() already checks round-trip flow, RANSAC and local
                # print appearance. Comparing all later frames with one fixed
                # root rejects valid perspective/lighting changes (014 f606).
                score = found['score']
                if n not in records or abs(n-seed) < abs(n-records[n]['root']):
                    records[n] = dict(frame=n, root=seed, quad=q.tolist(), score=score)
                previous = n
    for n, record in records.items():
        cv2.fillConvexPoly(result[n], np.rint(record['quad']).astype(np.int32), 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, alpha=result, protected=np.zeros_like(result),
            source_sha256=np.array(item['source_sha256']), source_dimensions=item['dimensions'],
            mask_dimensions=[width,height], frames_half_open=[0,count])
    dependencies = {p:project.digest(project.project_path(p)) for p in
                    (root['masks'], root['segmentation_import'], 'scripts/tracked_protection.py', 'scripts/carrier_tracking.py')}
    report = dict(file_id=item['id'], operation_id=spec['operation_id'],
        source_sha256=item['source_sha256'], backend='carrier_context_lk_ecc', root=root,
        dependencies=dependencies, seed_frames=seeds, masks=out, masks_sha256=project.digest(path),
        frames_half_open=[0,count], records=[records[n] for n in sorted(records)],
        status='prepared_pending_visual_review', accepted=False)
    path.with_suffix('.npz.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return dict(masks=out, tracked_frames=len(records), root_frames=seeds, accepted=False)


def validate(case, item, plan):
    path = project.project_path(plan['segmentation_import'])
    if project.digest(path) != plan['segmentation_import_sha256']:
        raise ValueError('Tracked protection evidence changed')
    report = project.read_json(path)
    if (report['backend'] != plan['segmentation_backend'] or report['file_id'] != item['id'] or
            report['operation_id'] != plan['operation_id'] or report['source_sha256'] != item['source_sha256'] or
            report['frames_half_open'] != plan['edit_frames_half_open'] or report['masks'] != plan['masks'] or
            report['masks_sha256'] != project.digest(project.project_path(plan['masks']))):
        raise ValueError('Tracked protection source or artifact changed')
    for dependency, expected in report['dependencies'].items():
        if project.digest(project.project_path(dependency)) != expected:
            raise ValueError('Tracked protection dependency changed')
    video_segmentation.validate_import(case,item,dict(report['root'],operation_id=plan['operation_id']))
