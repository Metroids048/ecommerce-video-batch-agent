"""Supervised seed, full-frame logo localization and foreground-safe ink masks. Offline only."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import project


def glyph_templates(seed_frame, seed_box):
    x, y, w, h = seed_box
    seed = cv2.cvtColor(seed_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    count, _, stats, _ = cv2.connectedComponentsWithStats((seed > 135).astype(np.uint8), 8)
    return [(int(xx), int(yy), seed[max(0, yy-1):yy+hh+1, max(0, xx-1):xx+ww+1])
            for xx, yy, ww, hh, area in stats[1:count] if area >= 20]


def refine_box(frame, box, glyphs):
    x, y, w, h = box
    x0, y0 = max(0, x-40), max(0, y-25)
    roi = cv2.cvtColor(frame[y0:y+h+25, x0:x+w+40], cv2.COLOR_BGR2GRAY)
    anchors = []
    for xx, yy, template in glyphs:
        _, score, _, pos = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
        if score > .8:
            anchors.append((float(score), x0+pos[0]-xx+1, y0+pos[1]-yy+1))
    if anchors:
        best = max(anchors)
        agree = [a for a in anchors if abs(a[1]-best[1]) < 6 and abs(a[2]-best[2]) < 6]
        x, y = np.rint(np.median(np.array(agree)[:, 1:], axis=0)).astype(int)
        w, h = 102, 28
    return [int(x), int(y), int(w), int(h)], anchors


def ink_mask(frame, box, cloth_polygons=None):
    """Continuous donor patch, with source foreground retained and supervised product boundaries."""
    x, y, w, h = box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = frame[:, :, 0].astype(np.int16)-frame[:, :, 2]
    # Chroma is smeared into white print by the source's subsampled encoding.
    # A weak warm tint alone is not evidence of an opaque finger.
    white_core = ((gray > 210) & (hsv[:, :, 1] < 45)).astype(np.uint8)
    white_print = (gray > 160) & (hsv[:, :, 1] < 75) & (cv2.dilate(white_core, np.ones((3, 3), np.uint8)) > 0)
    foreground = ((blue < -17) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 45) & ~white_print) | ((hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 75))
    region = np.zeros(frame.shape[:2], np.uint8)
    cv2.rectangle(region, (x-4, y-4), (x+w+4, y+h+4), 255, -1)
    alpha = cv2.GaussianBlur(region, (5, 5), 0)
    if cloth_polygons is not None:
        allowed = np.zeros_like(region)
        for polygon in cloth_polygons:
            cv2.fillPoly(allowed, [np.array(polygon, np.int32)], 255)
        alpha[allowed == 0] = 0
    alpha[foreground] = 0
    # Do not edit a fully covered target just because its predicted box exists.
    visible = (alpha > 0) & (gray > 110) & (hsv[:, :, 1] < 75) & ~foreground
    if not np.any(visible):
        alpha[:] = 0
    return alpha, foreground.astype(np.uint8)*255, []


def prepare(case, spec, out):
    project.validate_case(case)
    item = next(f for f in case['files'] if f['id'] == spec['file_id'])
    if item['source_sha256'] != spec['source_sha256']:
        raise ValueError('Seed belongs to another source')
    out = project.project_path(out)
    if out.exists():
        raise ValueError('Preserve existing track version')
    cap = cv2.VideoCapture(str(project.project_path(item['source'])))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    cap.set(1, spec['seed_frame'])
    ok, frame = cap.read()
    if not ok:
        raise ValueError('Cannot decode seed')
    x, y, w, h = spec['seed_box']
    template = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    glyphs = glyph_templates(frame, spec['seed_box'])
    bank = []
    for scale in spec['scales']:
        small = cv2.resize(template, None, fx=scale, fy=scale)
        for deg in spec['rotations_degrees']:
            hh, ww = small.shape
            matrix = cv2.getRotationMatrix2D((ww/2, hh/2), deg, 1)
            bank.append(cv2.warpAffine(small, matrix, (ww, hh), borderMode=cv2.BORDER_REPLICATE))
    records = []
    if spec.get('tracking_parent'):
        parent_path = project.project_path(spec['tracking_parent'])
        if project.digest(parent_path) != spec['tracking_parent_sha256']:
            raise ValueError('Tracking parent changed')
        parent = project.read_json(parent_path)
        if parent['source_sha256'] != item['source_sha256'] or parent['source_frame_count'] != count:
            raise ValueError('Tracking parent belongs to another input')
        records = [{k: r[k] for k in ('frame', 'confidence', 'detected_box')} for r in parent['review']['records']]
    else:
        cap.set(1, 0)
        sx, sy, sw, sh = spec['search_box']
        for n in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Source ended before its stated full frame count')
            gray = cv2.cvtColor(frame[sy:sy+sh, sx:sx+sw], cv2.COLOR_BGR2GRAY)
            best = (-1, None)
            for t in bank:
                _, value, _, pos = cv2.minMaxLoc(cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED))
                if value > best[0]:
                    best = (value, [pos[0]+sx, pos[1]+sy, t.shape[1], t.shape[0]])
            records.append({'frame': n, 'confidence': round(best[0], 4), 'detected_box': best[1]})
    boxes = np.array([r['detected_box'] for r in records], float)
    valid = np.array([r['frame'] for r in records if r['confidence'] >= .75])
    if len(valid) < 2:
        raise ValueError('Insufficient unobstructed target anchors')
    predicted = np.stack([np.interp(np.arange(count), valid, boxes[valid, k]) for k in range(4)], 1)
    masks = np.zeros((count, height, width), np.uint8)
    protected = np.zeros_like(masks)
    warps = np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))
    donors = np.full(count, spec['donor_frame'], np.int32)
    dx, dy, dw, dh = spec['donor_box_after_resize']
    cap.set(1, 0)
    for n, record in enumerate(records):
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Source decode failed during masks')
        b = boxes[n] if record['confidence'] >= .65 and np.max(np.abs(boxes[n, :2]-predicted[n, :2])) < 25 else predicted[n]
        box = np.rint(b).astype(int).tolist()
        box, anchors = refine_box(frame, box, glyphs)
        box = spec.get('box_corrections', {}).get(str(n), box)
        polygons = spec.get('cloth_polygons', {}).get(str(n))
        masks[n], protected[n], components = ink_mask(frame, box, polygons)
        if any(a <= n < b for a, b in spec.get('fully_occluded_intervals', [])):
            masks[n][:] = 0
        record['glyph_anchors'] = anchors
        for correction in spec.get('protected_ellipses', []):
            if correction['frames_half_open'][0] <= n < correction['frames_half_open'][1]:
                cv2.ellipse(protected[n], tuple(correction['center']), tuple(correction['axes']), correction.get('angle', 0), 0, 360, 255, -1)
        masks[n][protected[n] > 0] = 0
        x, y, w, h = box
        warps[n, 0, 0], warps[n, 1, 1] = w/dw, h/dh
        warps[n, 0, 2], warps[n, 1, 2] = x-dx*w/dw, y-dy*h/dh
        record.update(box=box, components=components, mask_pixels=int(np.count_nonzero(masks[n])),
                      state='visible' if record['confidence'] >= .75 else ('partial' if np.any(masks[n]) else 'occluded'))
        if not np.any(masks[n]) and record['confidence'] >= .75:
            record['state'] = 'unresolved'
    cap.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    mask_path = out.with_suffix('.npz')
    np.savez_compressed(mask_path, alpha=masks, protected=protected, raw_to_source=warps, donor_frames=donors)
    plan = {'case_version': case['case_version'], 'file_id': item['id'], 'source_sha256': item['source_sha256'],
            'source_frame_count': count, 'source_frames_half_open': [0, count],
            'cloud': spec['cloud'], 'cloud_sha256': project.digest(project.project_path(spec['cloud'])),
            'masks': str(mask_path.relative_to(project.ROOT)), 'masks_sha256': project.digest(mask_path),
            'parent_version': spec['parent_version'], 'provider_task_id': spec['provider_task_id'],
            'sampling_mode': 'cached_fabric_donor', 'full_scope_requested': True, 'encoding_crf': 16,
            'color_match': 'context_median', 'spec': spec,
            'review': {'method': 'supervised word seed; full-source localization; partial-glyph anchors; continuous donor cloth patch; source foreground protection; per-frame supervised product boundaries; visual review still required',
                       'frame_states': [r['state'] for r in records], 'records': records,
                       'full_mask_reviewed': False, 'automatic_generalization_proven': False}}
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    return {'plan': str(out.relative_to(project.ROOT)), 'frames': count,
            'masked_frames': sum(r['mask_pixels'] > 0 for r in records),
            'unresolved_frames': [r['frame'] for r in records if r['state'] == 'unresolved']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(project.read_json(project.project_path(args.case)), project.read_json(project.project_path(args.spec)), args.out), ensure_ascii=False))
