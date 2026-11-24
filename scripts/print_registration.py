"""Register complete source-observed print inside an independently proposed carrier."""
import base64
import zlib
import cv2
import numpy as np
from surface_text import ink_signal


def freeze_profile(reference,quad,size=(284,66)):
    plane=np.float32([[0,0],[size[0]-1,0],[size[0]-1,size[1]-1],[0,size[1]-1]])
    patch=cv2.warpPerspective(reference,cv2.getPerspectiveTransform(np.float32(quad),plane),size)
    return {'size':list(size),'pixels':base64.b64encode(zlib.compress(patch.tobytes())).decode('ascii')}


def profile_image(profile):
    w,h=profile['size']
    return np.frombuffer(zlib.decompress(base64.b64decode(profile['pixels'])),np.uint8).reshape(h,w,3)


def register_extent(profile,current,proposal,polarity):
    """Bounded full-glyph search; never authorize a substring or a blur-only fit."""
    from text_localization import expand_word
    context=expand_word(proposal,1.8,3)
    plane=np.float32([[0,0],[511,0],[511,199],[0,199]])
    matrix=cv2.getPerspectiveTransform(context,plane)
    patch=cv2.warpPerspective(current,matrix,(512,200))
    signal=ink_signal(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),polarity)
    source=ink_signal(cv2.cvtColor(profile_image(profile),cv2.COLOR_BGR2GRAY),polarity)
    # Search coordinates depend on the proposal expansion, not the stored
    # profile resolution. Otherwise a nondefault profile changes object size.
    source_width,source_height=round(511/1.8),round(199/3)
    best=None
    for sx in (.85,1.,1.15):
        for sy in (.75,1.,1.25):
            template=cv2.resize(source,(round(source_width*sx),round(source_height*sy)))
            template=cv2.GaussianBlur(template,(0,0),2)
            scores=cv2.matchTemplate(cv2.GaussianBlur(signal,(0,0),2),template,cv2.TM_CCOEFF_NORMED)
            _,score,_,point=cv2.minMaxLoc(scores)
            if best is None or score>best[0]:best=(score,point,template.shape)
    score,(x,y),(h,w)=best
    predicted=cv2.perspectiveTransform(np.float32([[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]]),np.linalg.inv(matrix))[0]
    return {'score':score,'word_quad':predicted.tolist(),'supported':score>=.7}


def fit_word_blur(profile,current,quad,polarity):
    from print_blur import fit_psf
    reference=profile_image(profile);h,w=reference.shape[:2]
    plane=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    matrix=cv2.getPerspectiveTransform(np.float32(quad),plane)
    observed=cv2.warpPerspective(current,matrix,(w,h))
    valid=cv2.warpPerspective(np.full(current.shape[:2],255,np.uint8),matrix,(w,h))>250
    def detrended(image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
        yy,xx=np.mgrid[:h,:w]
        basis=np.stack([np.ones_like(xx),xx/w,yy/h],axis=-1)
        border=((xx<w*.04)|(xx>w*.96)|(yy<h*.04)|(yy>h*.96))&valid
        if border.sum()<20:return None
        background=basis@np.linalg.lstsq(basis[border],gray[border],rcond=None)[0]
        return np.float32(gray-background)
    a,b=detrended(reference),detrended(observed)
    fitted=fit_psf(a,b,valid) if a is not None and b is not None else None
    if fitted and fitted['score']>=.7:
        fitted['signal_method']='linear_border_detrend'
        return fitted
    fitted=fit_psf(ink_signal(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),polarity),
                   ink_signal(cv2.cvtColor(observed,cv2.COLOR_BGR2GRAY),polarity),valid)
    if fitted:fitted['signal_method']='morphological_ink_fallback'
    return fitted
