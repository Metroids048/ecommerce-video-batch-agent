"""Experimental endpoint-flow reconstruction of baked crossfades.

Returns an actual pixel candidate and fit residual. A low residual does not
prove layer identity or occlusion recovery; this is not a production fallback.
"""
import cv2
import numpy as np


def warp_endpoint(current,endpoint,edited,support):
    size=(current.shape[1]//2,current.shape[0]//2)
    a=cv2.resize(cv2.cvtColor(current,cv2.COLOR_BGR2GRAY),size)
    b=cv2.resize(cv2.cvtColor(endpoint,cv2.COLOR_BGR2GRAY),size)
    flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a,b,None)
    flow=cv2.resize(flow,(current.shape[1],current.shape[0]))*2
    yy,xx=np.mgrid[:current.shape[0],:current.shape[1]]
    maps=(np.float32(xx+flow[:,:,0]),np.float32(yy+flow[:,:,1]))
    original=cv2.remap(endpoint,*maps,cv2.INTER_LINEAR)
    delta=cv2.remap(edited.astype(np.float32)-endpoint,*maps,cv2.INTER_LINEAR)
    mask=cv2.remap(np.uint8(support)*255,*maps,cv2.INTER_LINEAR)>250
    delta[~mask]=0
    return original,delta,mask


def reconstruct(current,first,last):
    a,da,ma=warp_endpoint(current,*first);b,db,mb=warp_endpoint(current,*last)
    selected=~cv2.dilate(np.uint8(ma|mb),np.ones((15,15),np.uint8)).astype(bool)
    difference=a.astype(float)-b;target=current.astype(float)-b
    denominator=float(np.square(difference[selected]).sum())
    alpha=float(np.clip((difference[selected]*target[selected]).sum()/max(1,denominator),0,1))
    fitted=alpha*a+(1-alpha)*b
    out=np.uint8(np.clip(np.rint(current+alpha*da+(1-alpha)*db),0,255))
    return out,dict(first_weight=alpha,unprinted_fit_mae=float(np.abs(fitted-current)[selected].mean()),
                    support_pixels=int((ma|mb).sum()),scope='experimental endpoint-flow pixel reconstruction; not accepted')
