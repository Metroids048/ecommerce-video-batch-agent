"""Rasterize Agent-observed source polygons; never infer object identity from colour."""
import json
import cv2
import numpy as np
import project
import workflow


def dimensions(value):
    if (not isinstance(value, list) or len(value) != 2 or
            any(type(v) is not int or v <= 0 for v in value)):
        raise ValueError('Explicit positive [width,height] dimensions required')
    return value


def polygon_union(polygons, coordinate_size, mask_size):
    width, height = mask_size
    mask = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or
                not np.isfinite(points).all() or np.any(points < 0) or
                np.any(points > coordinate_size)):
            raise ValueError('Polygon outside declared coordinate grid')
        scaled = np.rint(points * (np.array(mask_size)/coordinate_size)).astype(np.int32)
        # One fill per contour: fillPoly(all_contours) uses even-odd filling and
        # cuts holes in overlaps between independent targets (fresh-013 failure).
        cv2.fillPoly(mask, [scaled], 255)
    return mask


def rasterize(spec, source_size):
    source_size = dimensions(source_size)
    coordinates = dimensions(spec['coordinate_size'])
    size = dimensions(spec['mask_size'])
    for grid in (coordinates, size):
        if grid[0]*source_size[1] != grid[1]*source_size[0]:
            raise ValueError('Coordinate/mask aspect ratio differs from source')
    lo, hi = spec['frames_half_open']
    records = spec['records']
    if (type(lo) is not int or type(hi) is not int or not 0 <= lo < hi or
            [r['frame'] for r in records] != list(range(lo, hi))):
        raise ValueError('Declare exactly every frame in source order; do not infer missing frames')
    sigma = float(spec.get('feather_sigma', 0))
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Invalid feather sigma in mask pixels')
    alpha = np.zeros((hi-lo, size[1], size[0]), np.uint8)
    protected = np.zeros_like(alpha)
    for k, row in enumerate(records):
        state = row['state']
        targets = row['target_polygons']
        if state not in ('visible', 'absent', 'occluded') or bool(targets) != (state == 'visible'):
            raise ValueError('Visible frames require measured targets; absent/occluded frames require none')
        mask = polygon_union(targets, coordinates, size)
        guard = polygon_union(row['protected_polygons'], coordinates, size)
        if sigma:
            mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        mask[guard != 0] = 0  # Subtract after feathering, never bleed back onto skin/props.
        if state == 'visible' and not mask.any():
            raise ValueError('Visible target has no writable pixels after protection')
        alpha[k], protected[k] = mask, guard
    return alpha, protected


def prepare(case, spec, out):
    from production_effort import check
    check(case, spec['file_id'], start_method=True, method_family='local_recolor')
    if spec.get('method') == 'vitmatte_input':
        from video_matting import prepare_input
        return prepare_input(case,spec,out)
    if spec.get('method') == 'pymatting_closed_form':
        from alpha_refinement import prepare as refine_alpha
        return refine_alpha(case, spec, out)
    if spec.get('method') == 'carrier_context_lk_ecc':
        from tracked_protection import prepare as track
        return track(case, spec, out)
    item = next(f for f in case['files'] if f['id'] == spec['file_id'])
    operation = next(o for o in item['operations'] if o['id'] == spec['operation_id'])
    if case.get('execution_policy', {}).get('version') == 2 and (
            operation.get('kind') == 'recolor' or operation.get('surface_motion') == 'deformable'):
        raise ValueError('Deformable/recolor targets require provider video segmentation; per-frame polygon mattes are not an execution method')
    if (spec['source_sha256'] != item['source_sha256'] or
            project.digest(project.project_path(item['source'])) != item['source_sha256']):
        raise ValueError('Source matte belongs to another source')
    if spec['operation_id'] not in [op['id'] for op in item['operations']]:
        raise ValueError('Source matte operation does not belong to file')
    if spec['frames_half_open'][1] > item['source_frame_count_inventory']:
        raise ValueError('Source matte extends beyond original frames')
    output = project.project_path(out)
    sidecar = output.with_suffix(output.suffix+'.json')
    if output.suffix != '.npz' or output.exists() or sidecar.exists():
        raise ValueError('Use a new .npz matte version; never overwrite prior evidence')
    alpha, protected = rasterize(spec, item['dimensions'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, alpha=alpha, protected=protected,
            source_sha256=np.array(item['source_sha256']), source_dimensions=item['dimensions'],
            mask_dimensions=spec['mask_size'], frames_half_open=spec['frames_half_open'])
    report = dict(file_id=item['id'], operation_id=spec['operation_id'],
        source_sha256=item['source_sha256'], source_dimensions=item['dimensions'],
        coordinate_size=spec['coordinate_size'], mask_size=spec['mask_size'],
        frames_half_open=spec['frames_half_open'], records=spec['records'],
        spec_sha256=workflow.hash_json(spec), masks=out,
        masks_sha256=project.digest(output), status='prepared_pending_visual_review', accepted=False,
        scope='Measured geometry only; wrong semantic polygons can still pass. Review every target/foreground boundary in the shared compositor preview.')
    sidecar.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def validate_metadata(arrays, source_sha256, source_size, frames_half_open):
    """New mattes carry their source/grid/range; legacy plans retain archived semantics."""
    if 'source_dimensions' not in arrays:
        return
    if (str(arrays['source_sha256']) != source_sha256 or
            list(arrays['source_dimensions']) != list(source_size) or
            list(arrays['frames_half_open']) != list(frames_half_open) or
            list(arrays['mask_dimensions']) != list(arrays['alpha'].shape[1:][::-1])):
        raise ValueError('Source matte source/grid/range metadata mismatch')
