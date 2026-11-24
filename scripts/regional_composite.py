"""Source-based regional writeback; explicit provider-pixel or exact-asset mode."""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def match_context_color(source, cloud, alpha, fabric_only=False):
    region = (alpha > 0).astype(np.uint8)
    ring = (cv2.dilate(region, np.ones((61 if fabric_only else 15,)*2, np.uint8)) != 0) & (region == 0)
    if fabric_only:
        # Hands can leave only a narrow slit of cloth inside the replacement box.
        # Original dark cloth there is valid context; the bright print is excluded.
        ring = cv2.dilate(region, np.ones((61, 61), np.uint8)) != 0
        hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
        other_hsv = cv2.cvtColor(cloud, cv2.COLOR_BGR2HSV)
        source_blue = source[:, :, 0].astype(np.int16)-source[:, :, 2]
        cloud_blue = cloud[:, :, 0].astype(np.int16)-cloud[:, :, 2]
        cloth = ((hsv[:, :, 2] < 100) & (hsv[:, :, 1] < 160) & (source_blue >= -12) &
                 (other_hsv[:, :, 2] < 140) & (cloud_blue >= 0))
        ring &= cloth
        if np.count_nonzero(ring) < 30:
            # A crossing hand can cover the entire 30px neighborhood. Use measured
            # cloth from this same frame, at most 80px away, only with ample support.
            wider = (cv2.dilate(region, np.ones((161, 161), np.uint8)) != 0) & cloth
            if np.count_nonzero(wider) >= 100:
                ring = wider
    if not np.any(region) or np.count_nonzero(ring) < 30:
        raise ValueError('Insufficient unedited context for color matching')
    offset = np.median(source[ring].astype(np.float32) - cloud[ring].astype(np.float32), axis=0)
    return np.clip(cloud.astype(np.float32) + offset, 0, 255).astype(np.uint8), offset


def compose_frame(source, cloud, alpha, protected=None):
    if source.shape != cloud.shape or alpha.shape != source.shape[:2]:
        raise ValueError('Frame/mask dimensions differ')
    effective = alpha.copy()
    if protected is not None:
        if protected.shape != alpha.shape:
            raise ValueError('Protection dimensions differ')
        effective[protected != 0] = 0
    weight = effective.astype(np.float32)[..., None] / 255
    output = np.rint(source * (1 - weight) + cloud * weight).astype(np.uint8)
    return output, effective


def resolve_color_context(samples, editable, max_distance=12):
    """Fill short occluded gaps from measured neighboring frames, with provenance."""
    values = np.asarray(samples, np.float32).copy()
    known = np.flatnonzero(np.isfinite(values).all(axis=1) & editable)
    borrowed = []
    for index in np.flatnonzero(editable & ~np.isfinite(values).all(axis=1)):
        nearby = known[np.abs(known - index) <= max_distance]
        if not len(nearby):
            raise ValueError(f'No measured cloth context within {max_distance} frames of {index}')
        left, right = nearby[nearby < index], nearby[nearby > index]
        support = [int(left[-1]), int(right[0])] if len(left) and len(right) else [int(nearby[np.argmin(np.abs(nearby-index))])]
        if len(support) == 2:
            fraction = (index-support[0])/(support[1]-support[0])
            values[index] = values[support[0]]*(1-fraction)+values[support[1]]*fraction
        else:
            values[index] = values[support[0]]
        borrowed.append({'source_frame': int(index), 'measured_context_frames': support})
    values[~editable] = 0
    return values, borrowed


def validate_timing(source_fps, expected, cloud_fps, cloud_count, mask_count):
    if abs(source_fps - cloud_fps) > 1e-6:
        raise ValueError('Cloud frame rate differs; retiming is forbidden')
    if expected != cloud_count or expected != mask_count:
        raise ValueError('Cloud/mask frame count differs; padding/truncation is forbidden')


def validate_donor_map(indices, expected, cloud_count):
    if indices.shape != (expected,) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Explicit integer donor index required for every source frame')
    if np.any(indices < 0) or np.any(indices >= cloud_count):
        raise ValueError('Cached donor index outside actual cloud video')


def validate_full_coverage(interval, count, states, nonempty):
    if interval != [0, count] or len(states) != count or len(nonempty) != count:
        raise ValueError('Full scope requires assessment of every source frame')
    if any(state not in ('visible', 'partial', 'occluded', 'absent') for state in states):
        raise ValueError('Unresolved visibility states remain')
    if any(state in ('visible', 'partial') and not edited for state, edited in zip(states, nonempty)):
        raise ValueError('Unprocessed visible target remains')


def stream_hash(ffmpeg, file, audio=False):
    command = [ffmpeg, '-v', 'error', '-threads', '2', '-i', str(file), '-map', '0:a:0' if audio else '0:v:0']
    command += ['-c', 'copy'] if audio else ['-fps_mode', 'passthrough']
    command += ['-f', 'framehash', '-']
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
    return [line.strip() for line in result.stdout.splitlines() if line and not line.startswith('#')]


def color_modes_for(plan, start, end):
    overrides=plan.get('color_match_overrides',{})
    if any(not key.isdigit() or not start <= int(key) < end for key in overrides):
        raise ValueError('Color override outside source interval')
    modes=[overrides.get(str(frame),plan.get('color_match','none')) for frame in range(start,end)]
    if any(mode not in ('none','context_median') for mode in modes):
        raise ValueError('Unsupported bounded color mode')
    return modes


def render(case, plan, output):
    import project
    from production_effort import require_preview
    require_preview(case, plan)
    if plan.get("schema") == "execution_manifest_v1":
        from execution_plan import render as render_manifest
        return render_manifest(case, plan, output)
    if any('instances' in r for r in plan.get('review',{}).get('records',[])):
        raise ValueError('Instance plans require the unified execution manifest; legacy composition cannot discard instances')
    metadata = project.validate_case(case)
    item = next(f for f in case['files'] if f['id'] == plan['file_id'])
    source_has_audio = metadata[item['id']]['audio_present']
    if plan['source_sha256'] != item['source_sha256'] or plan['case_version'] != case['case_version']:
        raise ValueError('Plan belongs to another input/version')
    source = project.project_path(item['source'])
    cloud = project.project_path(plan['cloud'])
    mask_path = project.project_path(plan['masks'])
    if project.digest(cloud) != plan['cloud_sha256'] or project.digest(mask_path) != plan['masks_sha256']:
        raise ValueError('Cloud/mask hash changed')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('ffmpeg unavailable')
    source_cap, cloud_cap = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(cloud))
    encoder = None
    try:
        source_fps = source_cap.get(cv2.CAP_PROP_FPS)
        source_count = int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(source_cap.get(3)), int(source_cap.get(4))
        start, end = plan['source_frames_half_open']
        if not 0 <= start < end <= source_count or source_count != plan['source_frame_count']:
            raise ValueError('Source frame range/count changed')
        with np.load(mask_path, allow_pickle=False) as arrays:
            masks, warps = arrays['alpha'], arrays['raw_to_source']
            donors = arrays['donor_frames'] if 'donor_frames' in arrays else None
            protected = arrays['protected'] if 'protected' in arrays else None
        donor_cache = {}
        asset_mode = plan.get('composition_mode') == 'exact_asset_on_source_label'
        plane_masks = asset_mode and plan.get('mask_space') == 'label_plane'
        asset = None
        carrier = {}
        carrier_styles = {}
        if asset_mode:
            from exact_asset import compose_label, compose_planar_text
            asset_path = project.project_path(plan['exact_asset'])
            if project.digest(asset_path) != plan['exact_asset_sha256']:
                raise ValueError('Exact replacement asset hash changed')
            operation = next((op for op in item['operations'] if op['id'] == plan.get('operation_id')), None)
            if operation is None or plan['exact_asset'] != operation.get('replacement_asset'):
                raise ValueError('Asset does not belong to this operation')
            asset = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
            carrier = {start+r['frame']: r['source_quad'] for r in plan['review']['records']
                       if 'source_quad' in r}
            carrier_styles = {start+r['frame']: r.get('label_style', {}) for r in plan['review']['records']}
            if len(masks) != end-start or any(start+i not in carrier for i, m in enumerate(masks) if np.any(m)):
                raise ValueError('Missing source carrier track; automatic extension is not implemented')
        elif plan.get('sampling_mode') == 'cached_fabric_donor':
            if donors is None or len(masks) != end-start:
                raise ValueError('Missing source-bound fabric donor mapping/masks')
            validate_donor_map(donors, end-start, int(cloud_cap.get(7)))
            # Only a spatial cloth patch is reused; source actions/timing are never taken from the donor.
            for index in np.unique(donors):
                cloud_cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, raw = cloud_cap.read()
                if not ok:
                    raise ValueError('Cached donor decode failed')
                donor_cache[int(index)] = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
        else:
            if donors is not None:
                raise ValueError('Donor mapping requires explicit cached_fabric_donor mode')
            validate_timing(source_fps, end-start, cloud_cap.get(5), int(cloud_cap.get(7)), len(masks))
        expected_mask_shape = tuple(reversed(plan['label_style']['plane_size'])) if plane_masks else (height, width)
        if masks.shape[1:] != expected_mask_shape or warps.shape != (len(masks), 3, 3):
            raise ValueError('Mask/registration dimensions differ')
        if masks.dtype != np.uint8 or not np.isfinite(warps).all():
            raise ValueError('Invalid mask/registration values')
        if protected is not None and (protected.shape != masks.shape or protected.dtype != np.uint8):
            raise ValueError('Invalid foreground protection masks')
        if plan.get('full_scope_requested'):
            validate_full_coverage([start, end], source_count, plan['review']['frame_states'], np.any(masks, axis=(1, 2)))
        calibrated, borrowed_context = None, []
        color_modes=color_modes_for(plan,start,end)
        if donor_cache and 'context_median' in color_modes:
            samples = np.full((end-start, 3), np.nan, np.float32)
            editable = np.any(masks, axis=(1, 2)) & (np.asarray(color_modes)=='context_median')
            for index in range(source_count):
                ok, frame = source_cap.read()
                if not ok:
                    raise ValueError('Source decode failed during context preparation')
                if not start <= index < end or not editable[index-start]:
                    continue
                raw = cv2.warpPerspective(donor_cache[int(donors[index-start])], warps[index-start], (width, height))
                try:
                    _, samples[index-start] = match_context_color(frame, raw, masks[index-start], True)
                except ValueError:
                    pass  # A short gap must be resolved from measured context before encoding.
            calibrated, borrowed_context = resolve_color_context(samples, editable)
            source_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        crf = plan.get('encoding_crf', 18)
        if not isinstance(crf, int) or not 0 <= crf <= 51:
            raise ValueError('Invalid encoding quality')
        output = Path(output)
        if output.exists():
            raise ValueError('Output exists; preserve parent and choose a new version')
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-s', f'{width}x{height}', '-r', str(source_fps), '-i', '-',
                   '-i', str(source), '-map', '0:v:0', '-map', '1:a:0?',
                   '-c:v', 'libx264', '-preset', 'fast', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                   '-c:a', 'copy', '-video_track_timescale', '15360',
                   '-movie_timescale', '1000000', '-movflags', '+faststart', str(output)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        modified, changed_pixels, outside_changes, color_offsets = [], 0, 0, []
        n = 0
        while True:
            ok, frame = source_cap.read()
            if not ok:
                break
            if start <= n < end:
                mask = masks[n-start]
                protection = None if protected is None else protected[n-start]
                if plane_masks and np.any(mask):
                    pw, ph = plan['label_style']['plane_size']
                    plane = np.float32([[0, 0], [pw-1, 0], [pw-1, ph-1], [0, ph-1]])
                    projection = cv2.getPerspectiveTransform(plane, np.asarray(carrier[n], np.float32))
                    mask = cv2.warpPerspective(mask, projection, (width, height))
                    if protection is not None:
                        protection = cv2.warpPerspective(protection, projection, (width, height))
                elif plane_masks:
                    mask = np.zeros((height, width), np.uint8)
                    protection = None
                if asset_mode:
                    if not np.any(mask):
                        raw = frame
                    elif plane_masks:
                        raw = compose_planar_text(frame, carrier[n], asset, dict(plan['label_style'], **carrier_styles[n]))
                    else:
                        raw = compose_label(frame, carrier[n], asset)
                elif donor_cache:
                    raw = donor_cache[int(donors[n-start])]
                else:
                    raw_ok, raw = cloud_cap.read()
                    if not raw_ok:
                        raise ValueError('Cloud decode ended early; no padding allowed')
                    raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
                if not asset_mode:
                    raw = cv2.warpPerspective(raw, warps[n-start], (width, height), flags=cv2.INTER_LINEAR)
                if color_modes[n-start] == 'context_median' and np.any(masks[n-start]):
                    if calibrated is not None:
                        offset = calibrated[n-start]
                        raw = np.clip(raw.astype(np.float32)+offset, 0, 255).astype(np.uint8)
                    else:
                        try:
                            raw, offset = match_context_color(frame, raw, masks[n-start])
                        except ValueError as exc:
                            raise ValueError(f'Color context failed at source frame {n}: {exc}') from exc
                    color_offsets.append({'source_frame': n, 'bgr_offset': offset.tolist()})
                frame_out, effective = compose_frame(frame, raw, mask, protection)
                diff = np.any(frame_out != frame, axis=2)
                outside_changes += int(np.count_nonzero(diff & (effective == 0)))
                changed_pixels += int(np.count_nonzero(diff))
                if np.any(effective):
                    modified.append(n)
                frame = frame_out
            encoder.stdin.write(frame.tobytes())
            n += 1
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        encoder.stderr.close()
        code = encoder.wait(timeout=60)
        if code or n != source_count:
            raise ValueError(f'Encode/count failed: {code}, {n}/{source_count}: {error[-800:]}')
        if not asset_mode and not donor_cache and cloud_cap.read()[0]:
            raise ValueError('Extra cloud frames were not consumed')
    finally:
        source_cap.release()
        cloud_cap.release()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    # Packet hashes include PTS/DTS/duration and compressed AAC payload. No listening claim.
    output_has_audio = project.probe(output)['audio_present']
    audio_source = stream_hash(ffmpeg, source, True) if source_has_audio else []
    audio_output = stream_hash(ffmpeg, output, True) if output_has_audio else []
    video_source, video_output = stream_hash(ffmpeg, source), stream_hash(ffmpeg, output)
    timeline = lambda rows: [','.join(row.split(',')[:4]) for row in rows]
    report = {'file_id': item['id'], 'case_version': case['case_version'],
              'source_sha256': item['source_sha256'], 'cloud_sha256': plan['cloud_sha256'],
              'dependency': __import__('workflow').requirement_binding(case,item),
              'parent_version': plan['parent_version'], 'provider_task_id': plan['provider_task_id'],
              'output': str(output.relative_to(project.ROOT)), 'output_sha256': project.digest(output),
              'source_frames_half_open': [start, end], 'modified_source_frames': modified,
              'source_frame_count': len(video_source), 'output_frame_count': len(video_output),
              'timeline_exact': timeline(video_source) == timeline(video_output),
              'source_audio_packets': len(audio_source), 'output_audio_packets': len(audio_output),
              'audio_packets_pts_payload_exact': source_has_audio == output_has_audio and audio_source == audio_output,
              'audio_presence': {'source': source_has_audio, 'output': output_has_audio},
              'preencode_changes_outside_effective_mask': outside_changes,
              'preencode_changed_pixels': changed_pixels,
              'color_match': plan.get('color_match', 'none'), 'color_match_overrides':plan.get('color_match_overrides',{}), 'color_offsets': color_offsets,
              'cloth_context_policy': 'Measured dark fabric within 30px; same-frame 80px fallback requires at least 100 cloth pixels; temporal gaps limited to 12 frames',
              'temporally_borrowed_color_context': borrowed_context,
              'encoding': f'H264 CRF{crf}; lossy encoding can change decoded pixels outside mask',
              'sampling_mode': 'exact_asset_source_carrier' if asset_mode else plan.get('sampling_mode', 'same_time_cloud_frames'),
              'composition_mode': plan.get('composition_mode', 'provider_pixels'),
              'exact_asset_sha256': plan.get('exact_asset_sha256'),
              'provider_pixels_used': not asset_mode,
              'full_timeline_assessed': bool(plan.get('full_scope_requested')),
              'input_masks_review': plan['review'], 'full_target_coverage': False,
              'continuous_playback_reviewed': False, 'audio_listened': False,
              'status': ('full_length_candidate_pending_visual_review' if plan.get('full_scope_requested') else 'partial_edit_full_length_process_candidate'), 'accepted': False}
    if not report['timeline_exact'] or not report['audio_packets_pts_payload_exact'] or outside_changes:
        report['status'] = 'failed_integrity_check'
    return report
