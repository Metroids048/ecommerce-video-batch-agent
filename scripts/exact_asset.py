"""Finite, supervised label-text replacement on an existing source carrier track."""
import cv2
import numpy as np


def compose_planar_text(source, quad, asset, style, return_mask=False):
    """Clean old print on a uniform paper carrier, retaining its measured lighting.

    Occlusion is excluded by the source-bound effective mask in the compositor.
    This estimates the paper hidden by old glyphs; it is not model reconstruction.
    """
    width, height = style['plane_size']
    plane = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    quad = np.asarray(quad, np.float32)
    label = cv2.warpPerspective(source, cv2.getPerspectiveTransform(quad, plane), (width, height))
    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(gray, 71)
    delta = background.astype(np.int16)-gray if style['polarity'] == 'dark' else gray.astype(np.int16)-background
    glyphs = (delta > 6).astype(np.uint8)*255
    glyphs = cv2.dilate(glyphs, np.ones((5, 5), np.uint8))
    clean = cv2.inpaint(label, glyphs, 4, cv2.INPAINT_TELEA)
    support = np.full((height, width), 255, np.uint8)
    if style.get('paper_cleanup') == 'robust_border':
        # Fit the local illumination from clear margins, not a median filter
        # whose kernel can fit entirely inside a broad old letter.
        yy, xx = np.mgrid[:height, :width]
        basis = np.stack([np.ones_like(xx), xx/max(width-1, 1), yy/max(height-1, 1)], axis=-1)
        border = (xx < width*.06) | (xx >= width*.94) | (yy < height*.08) | (yy >= height*.92)
        values = label.astype(np.float64)
        selected = border.copy()
        for _ in range(4):
            coefficients = np.linalg.lstsq(basis[selected], values[selected], rcond=None)[0]
            background_rgb = np.clip(basis @ coefficients, 0, 255)
            residual = np.max(np.abs(values-background_rgb), axis=2)
            selected = border & (residual <= max(6, float(np.percentile(residual[border], 65))))
        contrast = np.mean(background_rgb-values, axis=2)
        if style['polarity'] == 'light':
            contrast = -contrast
        difference=background_rgb-values
        if style['polarity']=='light':difference=-difference
        center=(xx>width*.25)&(xx<width*.75)&(yy>height*.18)&(yy<height*.82)&(contrast>6)
        foreground=np.zeros((height,width),np.uint8)
        if center.sum()>20:
            vectors=difference[center]
            directions=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1)
            direction=np.median(directions,axis=0)
            direction/=max(np.linalg.norm(direction),1e-6)
            projection=difference@direction
            residual=np.linalg.norm(difference-projection[:,:,None]*direction,axis=2)
            # Out-of-palette foreground remains from the source, above both
            # cleanup and new ink. This is measured colour separation only;
            # same-colour occluders cannot be identified by this method.
            foreground=np.uint8((residual>np.maximum(9,np.abs(projection)*.28)) &
                                (np.linalg.norm(difference,axis=2)>12))*255
            foreground=cv2.morphologyEx(foreground,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            total,parts,stats,_=cv2.connectedComponentsWithStats(foreground)
            keep=[]
            boundary=np.unique(np.concatenate([parts[0],parts[-1],parts[:,0],parts[:,-1]]))
            for component in boundary:
                if component and stats[component,cv2.CC_STAT_AREA]>=width*height*.01:keep.append(component)
            foreground=np.uint8(np.isin(parts,keep))*255
            foreground=cv2.dilate(foreground,np.ones((3,3),np.uint8))
        # Smooth lighting and fabric gradients are not print. Morphological
        # background estimation removes broad strokes without turning a
        # nonlinear carrier gradient into a connected component at the edge.
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(height//2|1,height//2|1))
        if style['polarity']=='dark':
            ink_background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
            ink_delta=ink_background.astype(np.int16)-gray
        else:
            ink_background=cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
            ink_delta=gray.astype(np.int16)-ink_background
        minimum=style.get('minimum_ink_contrast',6)
        glyphs = np.uint8((ink_delta>minimum)&(contrast>minimum))*255
        glyphs[foreground!=0]=0
        # Neighboring rows can enter the outer context margin. Only components
        # connected to the recognized word are authorized for cleanup.
        _, components = cv2.connectedComponents(glyphs)
        left, top, right, bottom = style.get('cleanup_word_box', [.05, .12, .95, .88])
        word_components = np.unique(components[round(top*height):round(bottom*height),
                                              round(left*width):round(right*width)])
        word_components = word_components[word_components != 0]
        glyphs = np.uint8(np.isin(components, word_components))*255
        if np.any(glyphs[:, [0, -1]]) or np.any(glyphs[[0, -1], :]):
            raise ValueError('Old print intersects cleanup boundary; reacquire the complete word')
        glyphs = cv2.dilate(glyphs, np.ones((5, 5), np.uint8))
        clean = label.copy()
        clean[glyphs != 0] = np.rint(background_rgb[glyphs != 0]).astype(np.uint8)
        support = glyphs.copy()
    else:
        foreground=np.zeros((height,width),np.uint8)
    if style.get('paper_cleanup') == 'edge_interpolation':
        # Uniform paper only: interpolate measured clear margins. Foreground
        # writeback is still excluded by the independent protection mask.
        top = cv2.GaussianBlur(np.median(label[:6], axis=0).astype(np.float32)[None], (31, 1), 0)
        bottom = cv2.GaussianBlur(np.median(label[-6:], axis=0).astype(np.float32)[None], (31, 1), 0)
        fraction = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
        clean = np.rint(top*(1-fraction)+bottom*fraction).astype(np.uint8)
    yy, xx = np.nonzero(asset[:, :, 3])
    if not len(xx):
        raise ValueError('Empty exact asset')
    cropped = asset[yy.min():yy.max()+1, xx.min():xx.max()+1]
    fit_w, fit_h = style['asset_fit']
    scale = min(width*fit_w/cropped.shape[1], height*fit_h/cropped.shape[0])
    logo = cv2.resize(cropped, (round(cropped.shape[1]*scale), round(cropped.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    x, y = (width-logo.shape[1])//2, (height-logo.shape[0])//2
    opacity = style.get('ink_opacity', 1)
    if not 0 <= opacity <= 1:
        raise ValueError('Invalid ink opacity')
    alpha = logo[:, :, 3:4].astype(np.float32)/255 * opacity
    clean[y:y+logo.shape[0], x:x+logo.shape[1]] = np.rint(
        clean[y:y+logo.shape[0], x:x+logo.shape[1]]*(1-alpha)+logo[:, :, :3]*alpha).astype(np.uint8)
    support[y:y+logo.shape[0], x:x+logo.shape[1]] |= np.uint8(logo[:, :, 3] != 0)*255
    support[foreground!=0]=0
    if style.get('blur_sigma', 0):
        clean = cv2.GaussianBlur(clean, (0, 0), style['blur_sigma'])
    matrix = cv2.getPerspectiveTransform(plane, quad)
    size = (source.shape[1], source.shape[0])
    raw = cv2.warpPerspective(clean, matrix, size, borderMode=cv2.BORDER_REPLICATE)
    if return_mask:
        return raw, cv2.warpPerspective(support, matrix, size)
    return raw


def compose_label(source, quad, asset):
    """Keep original carrier. Clean bright old glyphs, then project the exact RGBA asset.

    This is local text cleanup, not provider reconstruction or texture preservation.
    Caller must bound the writeback mask inside the original stitching.
    """
    quad = np.asarray(quad, np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError('A verified four-point source carrier track is required')
    if asset.ndim != 3 or asset.shape[2] != 4 or not np.any(asset[:, :, 3]):
        raise ValueError('Exact asset must contain nonempty RGBA pixels')
    width, height = 256, 100
    plane = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    to_plane = cv2.getPerspectiveTransform(quad, plane)
    label = cv2.warpPerspective(source, to_plane, (width, height))
    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    # Only the inner print area; original seam and cloth border are outside it.
    zone = np.zeros((height, width), np.uint8)
    zone[23:77, 28:230] = 255
    local_background = cv2.medianBlur(gray, 51)
    glyphs = ((gray.astype(np.int16)-local_background > 3) & (zone != 0)).astype(np.uint8)*255
    glyphs = cv2.dilate(glyphs, np.ones((5, 5), np.uint8))
    glyphs[zone == 0] = 0
    clean = cv2.inpaint(label, glyphs, 5, cv2.INPAINT_TELEA)
    yy, xx = np.nonzero(asset[:, :, 3])
    cropped = asset[yy.min():yy.max()+1, xx.min():xx.max()+1]
    scale = min(166/cropped.shape[1], 30/cropped.shape[0])
    logo = cv2.resize(cropped, (round(cropped.shape[1]*scale), round(cropped.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    x, y = (width-logo.shape[1])//2, (height-logo.shape[0])//2
    alpha = logo[:, :, 3:4].astype(np.float32)/255
    clean[y:y+logo.shape[0], x:x+logo.shape[1]] = np.rint(
        clean[y:y+logo.shape[0], x:x+logo.shape[1]]*(1-alpha)+logo[:, :, :3]*alpha).astype(np.uint8)
    return cv2.warpPerspective(clean, cv2.getPerspectiveTransform(plane, quad), (source.shape[1], source.shape[0]))
