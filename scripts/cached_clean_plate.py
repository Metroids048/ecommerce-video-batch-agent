"""Project a source-bound static model clean plate, without retiming a video.

For rigid surfaces/background only. Per-frame source tracking and editable
mattes are explicit; this does not regenerate motion or imply raw-video pass.
"""
import cv2
import numpy as np
import project
import workflow
from regional_composite import compose_frame, validate_full_coverage


def validate(case,item,plan):
    if plan['dependency'] != workflow.requirement_binding(case,item):
        raise ValueError('Clean plate source/requirements changed')
    for path,digest in plan['artifact_hashes'].items():
        if project.digest(project.project_path(path)) != digest:
            raise ValueError('Clean plate artifact changed: '+path)
    origin=plan['plate_origin']
    if project.digest(project.project_path(origin['raw'])) != origin['raw_sha256']:
        raise ValueError('Original provider plate changed')
    if plan.get('exact_asset'):
        op=next(o for o in item['operations'] if o['id']==plan['operation_id'])
        if plan['exact_asset'] != op.get('replacement_asset'):
            raise ValueError('Clean plate wordmark belongs to another operation')
    with np.load(project.project_path(plan['masks']),allow_pickle=False) as z:
        count=plan['source_frame_count'];alpha=z['alpha'];matrices=z['plate_to_source']
        if alpha.dtype!=np.uint8 or alpha.ndim!=3 or matrices.shape!=(count,3,3) or not np.isfinite(matrices).all():
            raise ValueError('Invalid clean plate tracking/matte')
        validate_full_coverage(plan['source_frames_half_open'],count,plan['review']['frame_states'],
                               [bool(np.any(a)) for a in alpha])
    return plan['artifact_hashes']


class Layer:
    def __init__(self,plan,report):
        self.plan,self.report=plan,report
        self.plate=cv2.imread(str(project.project_path(plan['plate'])))
        if self.plate is None:raise ValueError('Cannot decode cached clean plate')
        with np.load(project.project_path(plan['masks']),allow_pickle=False) as z:
            self.alpha=z['alpha'];self.matrices=z['plate_to_source']
        report.update(method='cached_static_clean_plate_source_tracking',plate_origin=plan['plate_origin'],
                      regenerated_source_geometry=False,full_appearance_required=True)

    def compose(self,frame,n):
        h,w=frame.shape[:2];alpha=cv2.resize(self.alpha[n],(w,h),interpolation=cv2.INTER_LINEAR)
        if self.plan['review']['frame_states'][n] not in ('visible','partial'):
            return frame.copy(),np.zeros((h,w),bool)
        H=self.matrices[n]
        plate=cv2.warpPerspective(self.plate,H,(w,h),flags=cv2.INTER_LINEAR)
        valid=cv2.warpPerspective(np.full(self.plate.shape[:2],255,np.uint8),H,(w,h))
        if np.any((alpha>0)&(valid<255)):
            raise ValueError(f'Clean plate lacks support at source frame {n}')
        result,effective=compose_frame(frame,plate,alpha)
        touched=effective>0
        if not np.any(touched):raise ValueError(f'Visible target unwritten at source frame {n}')
        self.report['modified_source_frames'].append(n);self.report['modified_instance_count']+=1
        return result,touched
