"""Local carrier-context registration; rejects cuts/photometric mismatches."""
import cv2
import numpy as np
from text_localization import expand_word

def register(anchor,current,quad):
    found=feature_register(anchor,current,quad)
    return found if found is not None else patch_register(anchor,current,quad)


def feature_register(anchor,current,quad):
    q=np.float32(quad)
    mask=np.zeros(anchor.shape,np.uint8)
    cv2.fillConvexPoly(mask,np.int32(expand_word(q,1.7,2.4)),255)
    pts=cv2.goodFeaturesToTrack(anchor,250,.003,4,mask=mask)
    if pts is None or len(pts)<10:return None
    moved,s,_=cv2.calcOpticalFlowPyrLK(anchor,current,pts,None,winSize=(31,31),maxLevel=4)
    back,t,_=cv2.calcOpticalFlowPyrLK(current,anchor,moved,None,winSize=(31,31),maxLevel=4)
    good=(s.ravel()>0)&(t.ravel()>0)&(np.linalg.norm(pts-back,axis=2).ravel()<1.5)
    if good.sum()<10:return None
    for estimator in (cv2.estimateAffinePartial2D,cv2.estimateAffine2D):
        matrix,inliers=estimator(pts[good],moved[good],method=cv2.RANSAC,ransacReprojThreshold=1.5)
        if matrix is None or inliers.sum()<10:continue
        if estimator==cv2.estimateAffine2D:
            # Small inter-frame shear is real on the moving carrier. Keep the
            # unconstrained affine from collapsing a full word onto a substring.
            scales=np.linalg.svd(matrix[:,:2],compute_uv=False)
            if not (.8<min(scales)<=max(scales)<1.25 and max(scales)/min(scales)<1.3):continue
        pred=cv2.transform(q[None],matrix)[0]
        area=abs(cv2.contourArea(pred))/abs(cv2.contourArea(q))
        if not .4<area<2.5:continue
        score=appearance_score(anchor,current,q,pred)
        if score>=.6:return {'word_quad':pred,'score':score,'support':int(inliers.sum())}
    return None


def patch_register(anchor,current,quad):
    """Direct patch alignment when a small/blurred label has too few corners.

    Rigid motion in the rectified carrier prevents an unconstrained affine
    from shrinking the word onto a few surviving letters under an occluder.
    """
    context=expand_word(quad,1.8,2.5)
    plane=np.float32([[0,0],[383,0],[383,159],[0,159]])
    matrix=cv2.getPerspectiveTransform(np.float32(context),plane)
    a=cv2.warpPerspective(anchor,matrix,(384,160))
    b=cv2.warpPerspective(current,matrix,(384,160))
    if min(a.std(),b.std())<2:return None
    try:
        score,warp=cv2.findTransformECC(a,b,np.eye(2,3,dtype=np.float32),cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,1e-4),None,5)
    except cv2.error:return None
    pred=cv2.perspectiveTransform(cv2.transform(cv2.perspectiveTransform(
        np.float32(quad)[None],matrix),warp),np.linalg.inv(matrix))[0]
    verified=appearance_score(anchor,current,quad,pred)
    if score<.8 or verified<.6:return None
    return {'word_quad':pred,'score':verified,'support':0,'registration':'direct_patch_ECC'}



def appearance_score(anchor,current,quad,projected):
    plane=np.float32([[0,0],[255,0],[255,79],[0,79]])
    ma=cv2.getPerspectiveTransform(np.float32(quad),plane)
    mb=cv2.getPerspectiveTransform(np.float32(projected),plane)
    a=cv2.warpPerspective(anchor,ma,(256,80))
    b=cv2.warpPerspective(current,mb,(256,80))
    valid=(cv2.warpPerspective(np.full(anchor.shape,255,np.uint8),ma,(256,80))>250)&(cv2.warpPerspective(np.full(current.shape,255,np.uint8),mb,(256,80))>250)
    valid=cv2.erode(np.uint8(valid),np.ones((7,7),np.uint8))!=0
    if valid.mean()<.25:return -1.
    a=cv2.GaussianBlur(a,(0,0),2);b=cv2.GaussianBlur(b,(0,0),2)
    if min(a[valid].std(),b[valid].std())<2:return -1.
    score=float(np.corrcoef(a[valid],b[valid])[0,1])
    # Partial occlusion must retain independently matching portions of print.
    # Geometry comes from carrier registration, never from fitting a substring.
    tiles=[]
    for x in range(0,256,64):
        aa,bb=a[:,x:x+64],b[:,x:x+64]
        observed=valid[:,x:x+64]
        if observed.mean()>=.75 and min(aa[observed].std(),bb[observed].std())>=2:
            tiles.append(float(np.corrcoef(aa[observed],bb[observed])[0,1]))
    if len(tiles)>=2:score=max(score,min(sorted(tiles,reverse=True)[:2]))
    return score


def relocate(reference,quad,current):
    """Find a new carrier using surrounding print, then verify target identity."""
    from planar_text import carrier_instances
    from text_localization import match_reference
    sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.015)
    mask=np.zeros(reference.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,np.int32(expand_word(quad,1.8,3)),255)
    rk,rd=sift.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),mask)
    keys,desc=sift.detectAndCompute(cv2.cvtColor(current,cv2.COLOR_BGR2GRAY),None)
    if rd is None or desc is None:return []
    found=carrier_instances(rk,rd,quad,keys,desc,abs(cv2.contourArea(np.float32(quad))),
        current.shape[1],current.shape[0],minimum_spread=.2)
    return [dict(word_quad=i['source_quad'],registration_support=i['inliers'],
                 track_method='source_carrier_SIFT_verified',state='visible')
            for i in found if (match_reference(reference,quad,current,i['source_quad']) or
                               visible_print_match(reference,quad,current,i['source_quad']))]


def visible_print_match(reference,reference_quad,current,quad):
    """Compare only observed pixels when the geometric carrier leaves the image.

    Zero padding previously overwhelmed the print correlation and discarded a
    correct background carrier. It is not evidence about the offscreen print.
    """
    plane=np.float32([[0,0],[255,0],[255,79],[0,79]])
    m=cv2.getPerspectiveTransform(np.float32(quad),plane)
    valid=cv2.warpPerspective(np.full(current.shape[:2],255,np.uint8),m,(256,80))>250
    if valid.mean()<.25 or valid.mean()>.98:return False
    a=cv2.warpPerspective(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),
        cv2.getPerspectiveTransform(np.float32(reference_quad),plane),(256,80))
    b=cv2.warpPerspective(cv2.cvtColor(current,cv2.COLOR_BGR2GRAY),m,(256,80))
    # Keep the comparison away from interpolation across the image boundary.
    valid=cv2.erode(np.uint8(valid),np.ones((11,11),np.uint8))!=0
    # Compare local ink contrast, rather than the illumination field. A shadow
    # across the same carton previously dominated the correlation of luminance.
    from surface_text import ink_signal
    a=ink_signal(cv2.GaussianBlur(a,(0,0),1.5),'dark')
    b=ink_signal(cv2.GaussianBlur(b,(0,0),1.5),'dark')
    if min(a[valid].std(),b[valid].std())<3:return False
    return bool(np.corrcoef(a[valid],b[valid])[0,1]>.7)


def context_candidates(grays,records,scale,max_distance=30):
    """Bridge recognition gaps using two independent directions, root-checked.

    OCR is a source identity anchor, not a new text box shape on every frame.
    Predictions retain their root/roundtrip support; no interpolation over cuts.
    """
    candidates=[[]for _ in records]
    anchors=[(r['frame'],i)for r in records for i in r['instances']]
    for index,(seed,anchor) in enumerate(anchors):
        # Nearby roots are redundant, but every reappearance gets a fresh root.
        if index and seed-anchors[index-1][0]<4 and seed//4==anchors[index-1][0]//4:continue
        root=np.float32(anchor['word_quad'])*scale
        for direction in (-1,1):
            q=root.copy();previous=seed
            for distance in range(1,max_distance+1):
                n=seed+direction*distance
                if not 0<=n<len(grays):break
                found=register(grays[previous],grays[n],q)
                if found is None:break
                q=found['word_quad']
                score=appearance_score(grays[seed],grays[n],root,q)
                if score<.6:break
                candidates[n].append(dict(anchor,word_quad=(q/scale).tolist(),
                    instance_id=f"context-{seed}-{anchor['instance_id']}-{n}",
                    track_method='carrier_context_roundtrip_root_verified',anchor_frame=seed,
                    root_appearance_score=score,registration_support=found['support'],
                    root_distance=distance,state='visible'))
                previous=n
    return candidates
