"""Source-bound appearance references and native edit colour selection.

The native edit supplies a selection, never replacement texture or motion.
No optical flow, contour propagation, or frame-specific output patches.
"""
import cv2
import numpy as np
import project
import workflow


def references(case, item, recipe):
    manifest = project.read_json(project.project_path(recipe['reference_manifest']))
    if (project.digest(project.project_path(recipe['reference_manifest'])) != recipe['reference_manifest_sha256']
            or manifest['dependency'] != workflow.requirement_binding(case, item)):
        raise ValueError('Appearance reference source or manifest changed')
    images = recipe['provider_spec']['image_references']
    if not 1 <= len(images) <= 4 or images != [e['path'] for e in manifest['images']]:
        raise ValueError('Kling appearance references must match the frozen source examples')
    for entry in manifest['images']:
        path = project.project_path(entry['path'])
        image = cv2.imread(str(path))
        if (project.digest(path) != entry['sha256'] or image is None or path.stat().st_size > 10*1024*1024
                or image.shape[:2][::-1] != tuple(item['dimensions'])
                or not 0 <= entry['source_frame'] < item['source_frame_count_inventory']):
            raise ValueError('Appearance reference changed or has invalid source geometry')
    return {e['path']: e['sha256'] for e in manifest['images']}


def prepare_plan(case, item, ticket, plan, out):
    import runninghub_adapter as provider
    import source_timeline
    unit = provider.unit_for(project.read_json(workflow.state_path(case)), item['id'], ticket['recipe']['provider_spec']['unit_id'])
    request = provider.verify_unit(case, unit)
    raw = unit['raw']
    if project.digest(project.project_path(raw['path'])) != raw['sha256']:
        raise ValueError('Native guide output changed')
    timeline = source_timeline.require_cfr(source_timeline.build(project.project_path(raw['path'])))
    lo, hi = request['source_frames_half_open']
    if len(timeline['pts']) != hi-lo or timeline['fps'] != request['reference_fps']:
        raise ValueError('Native guide must preserve one output per reference frame')
    cap = cv2.VideoCapture(str(project.project_path(raw['path'])))
    masks = []
    try:
        for _ in range(hi-lo):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Incomplete native guide')
            hsv = cv2.cvtColor(cv2.resize(frame, tuple(item['dimensions']), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2HSV)
            # Broad green chroma band; object location and source material are
            # independently restricted by the existing source plan.
            masks.append(np.where((hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] >= 50), 255, 0).astype(np.uint8))
    finally:
        cap.release()
    path = out/'native-colour-guide.npz'
    np.savez_compressed(path, alpha=np.stack(masks))
    rel = str(path.relative_to(project.ROOT))
    plan['segmentation_protection'] = []
    plan['native_colour_guide'] = dict(masks=rel, source_frames_half_open=[lo,hi],
        unit_id=unit['unit_id'], request_fingerprint=unit['request_fingerprint'], raw=raw,
        dependency=request['dependency'])
    plan['artifact_hashes'][rel] = project.digest(path)
    return plan


def validate(case, item, plan):
    import runninghub_adapter as provider
    guide = plan['native_colour_guide']
    unit = provider.unit_for(project.read_json(workflow.state_path(case)), item['id'], guide['unit_id'])
    request = provider.verify_unit(case, unit)
    if (guide['dependency'] != workflow.requirement_binding(case, item)
            or unit['request_fingerprint'] != guide['request_fingerprint'] or unit['raw'] != guide['raw']
            or project.digest(project.project_path(guide['raw']['path'])) != guide['raw']['sha256']
            or guide['source_frames_half_open'] != request['source_frames_half_open']
            or project.digest(project.project_path(guide['masks'])) != plan['artifact_hashes'][guide['masks']]):
        raise ValueError('Native colour guide binding changed')
    with np.load(project.project_path(guide['masks']), allow_pickle=False) as z:
        alpha = z['alpha']
        lo, hi = guide['source_frames_half_open']
        if alpha.dtype != np.uint8 or alpha.shape != (hi-lo, item['dimensions'][1], item['dimensions'][0]):
            raise ValueError('Native colour guide source grid changed')
