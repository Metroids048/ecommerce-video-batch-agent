"""Local exemplar reconstruction experiments; no claim of true hidden fibers."""
import cv2
import numpy as np


def exemplar_inpaint(image,mask,patch_radius=7,donor_valid=None):
    """Copy measured texture from undamaged donors with local colour matching.

    Donors are frozen from the original visible surface, never synthesized
    pixels. No admissible donor means this method cannot reconstruct the hole.
    """
    radius=patch_radius;size=radius*2+1
    source=image.astype(np.float32);out=source.copy();unknown=mask!=0
    invalid=cv2.dilate(np.uint8(unknown),np.ones((size,size),np.uint8))!=0
    if donor_valid is not None:
        invalid|=cv2.erode(np.uint8(donor_valid),np.ones((size,size),np.uint8))==0
    invalid[:radius]=True;invalid[-radius:]=True;invalid[:,:radius]=True;invalid[:,-radius:]=True
    donors=np.argwhere(~invalid)
    if len(donors)<10:raise ValueError('No undamaged texture donor')
    # Deterministic bounded search, uniformly distributed over observed donors.
    donors=donors[::max(1,len(donors)//1200)]
    tiles=np.array([source[y-radius:y+radius+1,x-radius:x+radius+1]for y,x in donors])
    while unknown.any():
        known=np.float32(~unknown)
        support=cv2.boxFilter(known,-1,(size,size),normalize=False,borderType=cv2.BORDER_CONSTANT)
        frontier=unknown&(cv2.dilate(np.uint8(~unknown),np.ones((3,3),np.uint8))!=0)
        frontier[:radius]=False;frontier[-radius:]=False;frontier[:,:radius]=False;frontier[:,-radius:]=False
        if not frontier.any():raise ValueError('Hole reaches unobserved patch boundary')
        scores=np.where(frontier,support,-1);y,x=np.unravel_index(scores.argmax(),scores.shape)
        region=np.s_[y-radius:y+radius+1,x-radius:x+radius+1]
        observed=~unknown[region];target=out[region]
        delta=np.mean(target[observed],axis=0)-np.mean(tiles[:,observed,:],axis=1)
        differences=(tiles+delta[:,None,None,:]-target[None])[:,observed,:]
        costs=np.mean(differences**2,axis=(1,2))+np.mean(delta**2,axis=1)*.1
        best=int(costs.argmin());filled=tiles[best]+delta[best]
        # Fill one local patch; subsequent context includes reconstructed pixels,
        # but all sampled texture remains from originally observed donors.
        out[region][~observed]=filled[~observed];unknown[region]=False
    return np.uint8(np.clip(np.rint(out),0,255))


def reconstruct_weave(image,mask,donor_valid=None,illumination='pde'):
    """Separate interpolated illumination from measured exemplar grain."""
    base=cv2.inpaint(image,mask,3,cv2.INPAINT_NS)
    smooth=cv2.GaussianBlur(base,(0,0),3)
    if illumination=='surface_polynomial':
        # Fit only observed, undamaged material. This estimates smooth shading,
        # never claims to recover a hidden fold or printed-over real fibers.
        yy,xx=np.mgrid[:mask.shape[0],:mask.shape[1]]
        x=xx/max(1,mask.shape[1]-1);y=yy/max(1,mask.shape[0]-1)
        basis=np.stack([np.ones_like(x),x,y,x*x,x*y,y*y],axis=-1)
        known=cv2.dilate(mask,np.ones((17,17),np.uint8))==0
        if donor_valid is not None:known&=donor_valid
        if known.sum()<500:raise ValueError('Insufficient observed surface for shading fit')
        selected=known.copy()
        for _ in range(4):
            smooth=basis@np.linalg.lstsq(basis[selected],image[selected].astype(float),rcond=None)[0]
            residual=np.linalg.norm(image.astype(float)-smooth,axis=2)
            selected=known&(residual<=max(4,float(np.percentile(residual[known],75))))
    # Grain is only read from original undamaged support.
    grain=np.clip(image.astype(float)-cv2.GaussianBlur(image,(0,0),2).astype(float)+128,0,255).astype(np.uint8)
    expanded=cv2.dilate(mask,np.ones((9,9),np.uint8))
    filled=exemplar_inpaint(grain,expanded,donor_valid=donor_valid)
    out=image.copy();inside=mask!=0
    reconstructed=np.clip(smooth.astype(float)+filled.astype(float)-128,0,255)
    if illumination=='surface_polynomial':
        alpha=np.clip(cv2.distanceTransform(np.uint8(inside),cv2.DIST_L2,3)/3,0,1)[:,:,None]
        reconstructed=image*(1-alpha)+reconstructed*alpha
    out[inside]=np.uint8(np.clip(np.rint(reconstructed),0,255))[inside]
    return out
