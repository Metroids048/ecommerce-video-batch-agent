"""Local decoded presentation timeline; never estimates VFR frames from average fps."""
import bisect
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


def build(path):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('Local ffmpeg required for presentation timeline')
    # Inspect decoded AVFrame timestamps before an output encoder can quantize them.
    result = subprocess.run([ffmpeg, '-hide_banner', '-nostats', '-copyts', '-i', str(path),
                             '-map', '0:v:0', '-vf', 'showinfo=checksum=0',
                             '-fps_mode', 'passthrough', '-an', '-f', 'null', '-'],
                            capture_output=True, text=True, check=True, timeout=300)
    match = re.search(r'config in time_base: (\d+/\d+)', result.stderr)
    rows = re.findall(r' n:\s*(\d+) pts:\s*(-?\d+) pts_time:\S+\s+duration:\s*(\d+)', result.stderr)
    if not match or not rows:
        raise ValueError('No decoded presentation timestamps')
    if [int(row[0]) for row in rows] != list(range(len(rows))):
        raise ValueError('Non-contiguous decoded source frame index')
    pts = [int(row[1]) for row in rows]
    durations = [int(row[2]) for row in rows]
    duration_basis = 'decoded_frame_duration'
    # Some legitimate AVC files expose PTS but no AVFrame.duration. Recover a
    # CFR duration only when every measured step matches the declared rate.
    if durations and all(d == 0 for d in durations) and len(pts) > 1:
        rate = re.search(r'config in time_base: \d+/\d+, frame_rate: (\d+/\d+)', result.stderr)
        steps = [b-a for a,b in zip(pts,pts[1:])]
        if rate and len(set(steps)) == 1 and steps[0] > 0 and (
                Fraction(steps[0])*Fraction(match[1])*Fraction(rate[1]) == 1):
            durations = [steps[0]]*len(pts)
            duration_basis = 'uniform_decoded_pts_with_matching_declared_rate'
    if any(b <= a for a, b in zip(pts, pts[1:])) or any(d <= 0 for d in durations):
        raise ValueError('Non-monotonic or invalid source presentation timestamps')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return {'schema_version': 1, 'source_sha256': digest.hexdigest(),
            'time_base': match[1], 'pts': pts, 'durations': durations,
            'frame_count': len(pts), 'origin_pts': pts[0],
            'fps': str((Fraction(len(pts)-1, pts[-1]-pts[0]) / Fraction(match[1])).limit_denominator(1001)) if len(pts)>1 else str(1/(Fraction(durations[0])*Fraction(match[1]))),
            'duration_basis': duration_basis,
            'scope': 'decoded source presentation timestamps; not visual quality evidence'}


def frame_at_time(timeline, seconds):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        raise ValueError('Invalid source presentation time')
    pts, durations = timeline['pts'], timeline['durations']
    position = Fraction(str(seconds)) / Fraction(timeline['time_base']) + pts[0]
    # Browser/JSON seconds are floats: an exact displayed PTS can round a few
    # ulps below its tick. Snap only floating-point noise, never a frame interval.
    nearest = round(position)
    if abs(float(position-nearest)) <= max(4*math.ulp(float(position)), 1e-12):
        position = Fraction(nearest)
    if position < pts[0] or position >= pts[-1] + durations[-1]:
        raise ValueError('Feedback time outside source presentation timeline')
    return bisect.bisect_right(pts, position) - 1


def require_cfr(timeline):
    pts = timeline['pts']
    steps = [b-a for a, b in zip(pts, pts[1:])]
    # Quantized source times alternate by a tick for fractional frame periods.
    if steps and max(steps)-min(steps) > 1:
        raise ValueError('Variable frame rate source is unsupported by the current CFR exporter')
    if pts[0] != 0:
        raise ValueError('Nonzero source PTS origin is unsupported by the current CFR exporter')
    if steps and any(abs(duration - sum(steps)/len(steps)) > 1 for duration in timeline['durations']):
        raise ValueError('Variable frame durations are unsupported by the current CFR exporter')
    return timeline


def load_for_item(item):
    import project
    path = project.project_path(item['source_timeline'])
    if project.digest(path) != item['source_timeline_sha256']:
        raise ValueError('Source timeline index hash changed')
    timeline = json.loads(path.read_text())
    if timeline['source_sha256'] != item['source_sha256']:
        raise ValueError('Source timeline belongs to another input')
    return timeline
