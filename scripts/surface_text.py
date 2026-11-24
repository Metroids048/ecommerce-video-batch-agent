"""Local print inpainting and measured defocus, without generative reconstruction."""
import cv2
import numpy as np
import base64
import zlib


def source_ink_template(source,quad,size,polarity):
    """Freeze the observed old print, so clipped glyphs need not be rediscovered.

    Geometry is still supplied by the carrier tracker; this is not an image of
    the hidden background and cannot recover its true texture.
    """
    width,height=size
    plane=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    patch=cv2.warpPerspective(source,cv2.getPerspectiveTransform(np.float32(quad),plane),(width,height))
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(float)
    yy,xx=np.mgrid[:height,:width]
    basis=np.stack([np.ones_like(xx),xx/max(1,width-1),yy/max(1,height-1)],axis=-1)
    border=(xx<width*.04)|(xx>=width*.96)|(yy<height*.08)|(yy>=height*.92)
    selected=border.copy()
    for _ in range(4):
        background=basis@np.linalg.lstsq(basis[selected],gray[selected],rcond=None)[0]
        error=np.abs(gray-background)
        selected=border&(error<=max(4,float(np.percentile(error[border],65))))
    contrast=(background-gray) if polarity=='dark' else (gray-background)
    signal=np.maximum(contrast,ink_signal(gray.astype(np.float32),polarity))
    peak=float(np.percentile(signal,97))
    mask=np.uint8(signal>max(6,peak*.15))
    packed=base64.b64encode(zlib.compress(np.packbits(mask).tobytes())).decode('ascii')
    return dict(size=[width,height],bits=packed)


def compose_planar_text(source,quad,asset,style,return_mask=False):
    if 'asset_ink_bgr' in style:
        color=np.asarray(style['asset_ink_bgr'],dtype=float)
        if color.shape!=(3,) or not np.isfinite(color).all() or np.any((color<0)|(color>255)):
            raise ValueError('Invalid observed print color')
        asset=asset.copy()
        asset[:,:,:3]=np.rint(color).astype(np.uint8)
    if style.get('surface_cleanup')!='texture_inpaint':
        from exact_asset import compose_planar_text as original
        return original(source,quad,asset,style,return_mask)
    width,height=style['plane_size']
    plane=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    quad=np.float32(quad)
    rectification=cv2.getPerspectiveTransform(quad,plane)
    label=cv2.warpPerspective(source,rectification,(width,height))
    valid=cv2.warpPerspective(np.full(source.shape[:2],255,np.uint8),rectification,(width,height))>250
    # Offscreen pixels are unknown, never reflected copies of the source. A
    # nearest observed extension only supplies the local filter boundary; it is
    # excluded from statistics, repair support and the final write mask.
    if not valid.all():
        _,nearest=cv2.distanceTransformWithLabels(np.uint8(~valid),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        lookup=label[valid]
        label[~valid]=lookup[nearest[~valid]-1]
    clean,support,_=replace(label,asset,style,valid)
    matrix=cv2.getPerspectiveTransform(plane,quad)
    size=(source.shape[1],source.shape[0])
    raw=cv2.warpPerspective(clean,matrix,size,borderMode=cv2.BORDER_REPLICATE)
    mask=cv2.warpPerspective(support,matrix,size)
    return (raw,mask) if return_mask else raw


def ink_signal(gray, polarity):
    diameter=max(9,(gray.shape[0]//2)|1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter,diameter))
    if polarity=='light':
        return gray.astype(np.float32)-cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
    return cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel).astype(np.float32)-gray


def covariance_kernel(covariance):
    covariance=np.float64(covariance)
    values=np.linalg.eigvalsh(covariance)
    if not np.isfinite(covariance).all() or min(values)<=0 or max(values)>1000:
        raise ValueError('Invalid measured print blur covariance')
    radius=int(np.ceil(3*np.sqrt(max(values))))
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    points=np.stack([xx,yy],axis=-1)
    kernel=np.exp(-.5*np.einsum('...i,ij,...j->...',points,np.linalg.inv(covariance),points))
    return np.float32(kernel/kernel.sum())


def measured_blur(gray, polarity, valid=None):
    """Estimate edge spread on original ink, in rectified pixels (not frame IDs)."""
    contrast=ink_signal(gray,polarity)
    if valid is None:valid=np.ones(gray.shape,bool)
    peak=float(np.percentile(contrast[valid],98))
    if peak<8:return 1.
    gradient=cv2.magnitude(cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,1,0,ksize=3)/8,
                           cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,0,1,ksize=3)/8)
    edge=(contrast>peak*.25)&(contrast<peak*.75)&(gradient>1)&valid
    if edge.sum()<12:return 1.
    return float(np.clip(peak/(np.percentile(gradient[edge],70)*np.sqrt(2*np.pi)),.6,6))


def replace(label,asset,style,valid=None):
    """Inpaint only original ink; surrounding texture/seams are source pixels.

    shortcut: hidden fibers are locally interpolated; nonuniform weave under
    large opaque print requires a clean plate or a stronger reconstruction method.
    """
    height,width=label.shape[:2]
    if valid is None:valid=np.ones((height,width),bool)
    gray=cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
    signal=ink_signal(gray,style['polarity'])
    left,top,right,bottom=style.get('cleanup_word_box',[.06,.12,.94,.88])
    zone=np.zeros((height,width),np.uint8)
    zone[max(0,round(top*height)-5):min(height,round(bottom*height)+5),
         max(0,round(left*width)-round(width*.06)):min(width,round(right*width)+round(width*.06))]=255
    if style.get('cleanup_observed_polygon'):
        # Registration may tighten the new lettering, but cannot discard the
        # already observed old-print extent (including its motion trail).
        points=np.float32(style['cleanup_observed_polygon'])*[width-1,height-1]
        cv2.fillConvexPoly(zone,np.int32(np.rint(points)),255)
    zone[~valid]=0
    if np.count_nonzero(zone)<20:raise ValueError('Insufficient observed surface for repair')
    peak=float(np.percentile(signal[zone!=0],97))
    if peak<8:raise ValueError('Insufficient original ink contrast for surface repair')
    glyphs=np.uint8((signal>max(4,peak*.12))&(zone!=0))*255
    if style.get('source_ink_template'):
        template=style['source_ink_template'];tw,th=template['size']
        bits=np.frombuffer(zlib.decompress(base64.b64decode(template['bits'])),np.uint8)
        old=np.unpackbits(bits)[:tw*th].reshape(th,tw)*255
        x0,y0,x1,y1=round(left*width),round(top*height),round(right*width),round(bottom*height)
        glyphs[y0:y1,x0:x1]=np.maximum(glyphs[y0:y1,x0:x1],cv2.resize(old,(x1-x0,y1-y0)))
    glyphs=cv2.morphologyEx(glyphs,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    covariance=style.get('print_blur_covariance',style.get('directional_blur_covariance'))
    if covariance is not None:
        trail=covariance_kernel(covariance)
        glyphs=cv2.dilate(glyphs,np.uint8(trail>trail.max()*.02))
    sigma=measured_blur(gray,style['polarity'],valid)
    radius=max(2,int(np.ceil(sigma)))
    glyphs=cv2.dilate(glyphs,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2))
    # Local PDE interpolation, rather than flattening every old glyph to a plane.
    protection=foreground_protection(label,style['polarity'],valid)
    glyphs[protection!=0]=0
    clean=cv2.inpaint(label,np.maximum(glyphs,np.uint8(~valid)*255),3,cv2.INPAINT_NS)
    yy,xx=np.nonzero(asset[:,:,3])
    cropped=asset[yy.min():yy.max()+1,xx.min():xx.max()+1]
    # White print inherits measured illumination/transition contrast from the
    # original ink. A full-white overlay otherwise stays bright as the shot
    # fades, despite having the right geometry and edge blur.
    ink_rgb=np.ones(3,np.float32)*255
    if style['polarity']=='light' and np.min(cropped[:,:,:3][cropped[:,:,3]>0])>=240:
        observed=(zone!=0)&(protection==0)
        if np.count_nonzero(observed)>=20:
            samples=(signal>=max(8,float(np.percentile(signal[observed],95))))&observed
            if np.count_nonzero(samples)>=12:ink_rgb=np.median(label[samples],axis=0).astype(np.float32)
    fit_w,fit_h=style['asset_fit']
    scale=min(width*fit_w/cropped.shape[1],height*fit_h/cropped.shape[0])
    logo=cv2.resize(cropped,(max(1,round(cropped.shape[1]*scale)),max(1,round(cropped.shape[0]*scale))),interpolation=cv2.INTER_AREA)
    logo=logo.astype(np.float32)
    logo[:,:,:3]*=ink_rgb/255
    ink=np.zeros((height,width,4),np.float32)
    x,y=(width-logo.shape[1])//2,(height-logo.shape[0])//2
    a=logo[:,:,3:4].astype(np.float32)/255
    ink[y:y+logo.shape[0],x:x+logo.shape[1],:3]=logo[:,:,:3]*a
    ink[y:y+logo.shape[0],x:x+logo.shape[1],3:]=a
    if style.get('print_blur_covariance') is not None:
        # This is the total measured spread (reference plus fitted blur).
        # Adding the current edge-spread estimate would count blur twice.
        ink=cv2.filter2D(ink,-1,covariance_kernel(style['print_blur_covariance']))
    elif style.get('directional_blur_covariance'):
        legacy_covariance=np.float64(style['directional_blur_covariance'])+np.eye(2)*sigma**2
        ink=cv2.filter2D(ink,-1,covariance_kernel(legacy_covariance))
    else:
        ink=cv2.GaussianBlur(ink,(0,0),sigma)
    alpha=ink[:,:,3:4]
    out=np.rint(clean*(1-alpha)+ink[:,:,:3]).astype(np.uint8)
    support=np.maximum(glyphs,np.uint8(alpha[:,:,0]>.001)*255)
    support[protection!=0]=0
    support[~valid]=0
    out[support==0]=label[support==0]
    return out,support,{'measured_blur_sigma':sigma,'old_ink_contrast':peak}


def foreground_protection(label,polarity,valid=None):
    """Preserve boundary-connected colours inconsistent with the measured ink."""
    height,width=label.shape[:2]
    if valid is None:valid=np.ones((height,width),bool)
    yy,xx=np.mgrid[:height,:width]
    basis=np.stack([np.ones_like(xx),xx/max(1,width-1),yy/max(1,height-1)],axis=-1)
    border=((xx<width*.06)|(xx>=width*.94)|(yy<height*.08)|(yy>=height*.92))&valid
    if border.sum()<20:return np.zeros((height,width),np.uint8)
    selected=border.copy();values=label.astype(np.float64)
    # A cast shadow changes brightness without creating a foreground object.
    # Fit colour ratios so the shadow cannot protect the entire old word.
    values=values*128/np.maximum(values.mean(axis=2,keepdims=True),20)
    for _ in range(4):
        coefficients=np.linalg.lstsq(basis[selected],values[selected],rcond=None)[0]
        bg=np.clip(basis@coefficients,0,255)
        residual=np.max(np.abs(values-bg),axis=2)
        selected=border&(residual<=max(6,float(np.percentile(residual[border],65))))
    difference=bg-values
    if polarity=='light':difference=-difference
    signal=ink_signal(cv2.cvtColor(label,cv2.COLOR_BGR2GRAY),polarity)
    center=(xx>width*.25)&(xx<width*.75)&(yy>height*.2)&(yy<height*.8)&valid&(signal>max(6,float(np.percentile(signal[valid],85))))
    if center.sum()<20:return np.zeros((height,width),np.uint8)
    vectors=difference[center]
    direction=np.median(vectors/np.maximum(1,np.linalg.norm(vectors,axis=1,keepdims=True)),axis=0)
    direction/=max(np.linalg.norm(direction),1e-6)
    projection=difference@direction
    residual=np.linalg.norm(difference-projection[:,:,None]*direction,axis=2)
    fg=np.uint8((residual>np.maximum(9,np.abs(projection)*.28))&(np.linalg.norm(difference,axis=2)>12))*255
    if polarity=='light':
        # A dark sleeve over a light printed cloth label can have the same
        # colour ratios as the substrate. Its opposite luminance is additional
        # evidence; white print alone cannot explain that boundary component.
        gray=cv2.cvtColor(label,cv2.COLOR_BGR2GRAY).astype(float)
        selected=border.copy()
        for _ in range(4):
            luminance=basis@np.linalg.lstsq(basis[selected],gray[selected],rcond=None)[0]
            error=np.abs(gray-luminance)
            selected=border&(error<=max(6,float(np.percentile(error[border],65))))
        fg=np.maximum(fg,np.uint8((gray<luminance-25)&valid)*255)
    fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    _,parts,stats,_=cv2.connectedComponentsWithStats(fg)
    boundary=np.unique(np.concatenate([parts[0],parts[-1],parts[:,0],parts[:,-1]]))
    keep=[c for c in boundary if c and stats[c,cv2.CC_STAT_AREA]>=width*height*.01]
    foreground=np.uint8(np.isin(parts,keep))*255
    # Nails/highlights can have the carrier colour but lie inside the same
    # occluder silhouette. Retain those holes with the source foreground.
    contours,_=cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(foreground,contours,-1,255,cv2.FILLED)
    radius=max(2,round(height*.04))
    return cv2.dilate(foreground,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2))
