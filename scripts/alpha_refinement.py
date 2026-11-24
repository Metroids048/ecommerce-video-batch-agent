"""Closed-form alpha matting of cached semantic masks, through project.py matte.

The trimap comes from the source-bound target minus its semantic protections.
No per-frame painting, new segmentation submission, or regenerated video.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
import cv2
import numpy as np
import project
import workflow


def refine(frame, mask, width, radius, fixed_protection=None):
    from pymatting import estimate_alpha_cf
    height = round(frame.shape[0]*width/frame.shape[1])
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA).astype(np.float64)/255
    binary = (cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    fg, bg = cv2.erode(binary, kernel), 1-cv2.dilate(binary, kernel)
    if not fg.any():
        raise ValueError('Matting trimap has no confirmed foreground; revisit segmentation')
    if not bg.any():
        raise ValueError('Matting trimap has no confirmed background')
    trimap = np.full(binary.shape, .5)
    trimap[fg != 0], trimap[bg != 0] = 1., 0.
    if fixed_protection is not None:
        fixed = cv2.resize(fixed_protection, (width,height), interpolation=cv2.INTER_NEAREST) != 0
        fg[fixed], bg[fixed], trimap[fixed] = 0, 1, 0.
    alpha = estimate_alpha_cf(image, trimap, cg_kwargs={'maxiter': 300})
    if not np.isfinite(alpha).all() or np.any(alpha[bg != 0]) or not np.all(alpha[fg != 0] == 1):
        raise ValueError('Matting violated the known trimap constraints')
    return np.rint(alpha*255).astype(np.uint8)


def prepare(case, spec, out):
    import source_recolor
    sys.path.append(str(project.ROOT/'.runtime/matting'))
    from importlib.metadata import version
    if version('pymatting') != '1.1.14':
        raise ValueError('Use the pinned PyMatting runtime')
    item = workflow.item_for(case, spec['file_id'])
    plan_path = project.project_path(spec['root_plan'])
    plan = project.read_json(plan_path)
    if plan.get('alpha_refinement') or plan['operation_id'] != spec['operation_id']:
        raise ValueError('Refine the original semantic matte, never recursively refine a candidate')
    dependencies = source_recolor.validate(case, item, plan)
    dependencies.update({spec['root_plan']: project.digest(plan_path),
        item['source']: item['source_sha256'],
        'scripts/alpha_refinement.py': project.digest(project.ROOT/'scripts/alpha_refinement.py'),
        'scripts/source_recolor.py': project.digest(project.ROOT/'scripts/source_recolor.py')})
    width, radius = spec['width'], spec['unknown_radius']
    if type(width) is not int or not 256 <= width <= item['dimensions'][0] or type(radius) is not int or not 1 <= radius <= 32:
        raise ValueError('Explicit bounded matting resolution and unknown band required')
    output = project.project_path(out)
    sidecar = output.with_suffix('.npz.json')
    if output.suffix != '.npz' or output.exists() or sidecar.exists():
        raise ValueError('Preserve previous refined mattes')
    layer = source_recolor.Layer(plan, dict(modified_source_frames=[], modified_instance_count=0))
    cap = cv2.VideoCapture(str(project.project_path(item['source'])))
    lo, hi = plan['edit_frames_half_open']
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    result, batch = [], []
    at = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for n in range(lo, hi):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Matting source ended early')
                batch.append(pool.submit(refine, frame, layer.mask(frame, n), width, radius,
                    layer.fixed_protection(frame, n)))
                if len(batch) == 12 or n == hi-1:
                    result.extend(f.result() for f in batch)
                    batch.clear()
                    print(json.dumps(dict(stage='alpha_matting', frames=len(result), total=hi-lo)), flush=True)
    finally:
        cap.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, alpha=np.asarray(result), source_sha256=np.array(item['source_sha256']),
            source_dimensions=item['dimensions'], frames_half_open=[lo, hi])
    report = dict(method='pymatting_closed_form', version='1.1.14', file_id=item['id'],
        operation_id=plan['operation_id'], source_sha256=item['source_sha256'],
        root_plan=spec['root_plan'], dependencies=dependencies, masks=out,
        masks_sha256=project.digest(output), frames_half_open=[lo, hi],
        width=width, unknown_radius=radius, wall_seconds=time.monotonic()-at,
        status='prepared_pending_visual_review', accepted=False)
    sidecar.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def validate(case, item, plan):
    entry = plan['alpha_refinement']
    sidecar = project.project_path(entry['sidecar'])
    if project.digest(sidecar) != entry['sidecar_sha256']:
        raise ValueError('Alpha refinement evidence changed')
    report = project.read_json(sidecar)
    root = project.read_json(project.project_path(report['root_plan']))
    expected = dict(plan); expected.pop('alpha_refinement')
    if (report['method'] != entry['method'] or entry['method'] != 'pymatting_closed_form' or
            report['source_sha256'] != item['source_sha256'] or report['file_id'] != item['id'] or
            report['operation_id'] != plan['operation_id'] or root != expected or
            report['frames_half_open'] != plan['edit_frames_half_open'] or report['masks'] != entry['masks'] or
            project.digest(project.project_path(entry['masks'])) != report['masks_sha256']):
        raise ValueError('Alpha refinement belongs to another semantic matte or source')
    for path, expected_hash in report['dependencies'].items():
        if project.digest(project.project_path(path)) != expected_hash:
            raise ValueError('Alpha refinement dependency changed: '+path)
    return dict(report['dependencies'], **{entry['masks']: report['masks_sha256'], entry['sidecar']: entry['sidecar_sha256']})
