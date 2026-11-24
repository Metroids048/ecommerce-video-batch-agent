"""Estimate a source print PSF from a registered reference, in label coordinates."""
import cv2
import numpy as np


def gaussian_kernel(major,minor,angle):
    radius=max(1,int(np.ceil(major*3)))
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    theta=np.deg2rad(angle)
    along=xx*np.cos(theta)+yy*np.sin(theta)
    across=-xx*np.sin(theta)+yy*np.cos(theta)
    kernel=np.exp(-.5*((along/max(.4,major))**2+(across/max(.4,minor))**2))
    return np.float32(kernel/kernel.sum())


def fit_psf(reference,observed,valid=None):
    """Fit bounded anisotropic spread; keep ambiguity and fit error explicit.

    Both inputs contain the same registered old print, with background removed.
    A poor shape match is not evidence of a motion direction.
    """
    reference=np.float32(reference);observed=np.float32(observed)
    scale=min(1.,160/reference.shape[1])
    size=(round(reference.shape[1]*scale),round(reference.shape[0]*scale))
    ref=cv2.resize(reference,size,interpolation=cv2.INTER_AREA)
    obs=cv2.resize(observed,size,interpolation=cv2.INTER_AREA)
    valid=np.ones(obs.shape,bool) if valid is None else cv2.resize(np.uint8(valid),size,interpolation=cv2.INTER_NEAREST)!=0
    if valid.sum()<40 or min(ref[valid].std(),obs[valid].std())<1:
        return None
    # Allow a small registration translation, never reshape a word to fit blur.
    best=None;isotropic=None
    for major in (.4,.8,1.4,2.2,3.2,4.5):
        for minor in sorted(set((.4,min(1.,major),major))):
            for angle in ((0,) if minor==major else range(0,180,30)):
                prediction=cv2.filter2D(ref,-1,gaussian_kernel(major,minor,angle),borderType=cv2.BORDER_REPLICATE)
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        moved=cv2.warpAffine(prediction,np.float32([[1,0,dx],[0,1,dy]]),size,borderMode=cv2.BORDER_REPLICATE)
                        a=moved[valid];b=obs[valid]
                        score=float(np.corrcoef(a,b)[0,1])
                        row=dict(score=score,major_sigma=major/scale,minor_sigma=minor/scale,angle_degrees=angle,translation=[dx/scale,dy/scale])
                        if best is None or score>best['score']:best=row
                        if major==minor and (isotropic is None or score>isotropic['score']):isotropic=row
    best['isotropic_score']=isotropic['score']
    best['isotropic_sigma']=isotropic['major_sigma']
    best['direction_supported']=bool(best['score']>=.7 and best['score']-isotropic['score']>.015 and best['major_sigma']>best['minor_sigma']*1.5)
    return best
