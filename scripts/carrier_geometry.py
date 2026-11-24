"""Registration on observed carrier texture, independently of printed glyphs."""
import cv2
import numpy as np


def dark_surface_proposals(frame):
    """Multiscale dark carrier regions, before reading any glyphs or old tracks."""
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);height,width=gray.shape;found=[]
    scale=width/720
    for kw,kh in ((61,31),(121,61),(241,121)):
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(max(3,round(kw*scale)),max(3,round(kh*scale))))
        signal=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
        for threshold in (10,20,35):
            mask=cv2.morphologyEx(np.uint8(signal>threshold)*255,cv2.MORPH_CLOSE,np.ones((5,9),np.uint8))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour)<1500*scale**2:continue
                box=cv2.boxPoints(cv2.minAreaRect(contour));box=box[np.argsort(box[:,1])]
                top=box[:2][np.argsort(box[:2,0])];bottom=box[2:][np.argsort(box[2:,0])]
                q=np.float32([top[0],top[1],bottom[1],bottom[0]])
                if cv2.isContourConvex(q):
                    found.append(dict(quad=[[float(x/width),float(1-y/height)]for x,y in q]))
    return found


def relocate_rectangles(frame, rectangles, reference_carrier, reference_word, profile, polarity):
    """Independent full-frame proposals, verified against the complete source print.

    Rectangle confidence alone is never target identity. The carrier gives the
    search geometry even when OCR has not returned a word.
    """
    from print_registration import register_extent
    unit=np.float32([[0,0],[1,0],[1,1],[0,1]])
    uv=cv2.perspectiveTransform(np.float32(reference_word)[None],
        cv2.getPerspectiveTransform(np.float32(reference_carrier),unit))[0]
    height,width=frame.shape[:2];found=[]
    for rectangle in rectangles:
        q=np.float32([[x*width,(1-y)*height]for x,y in rectangle['quad']])
        ratio=np.linalg.norm(q[1]-q[0])/max(1,np.linalg.norm(q[2]-q[1]))
        if abs(cv2.contourArea(q))<1500*(width/720)**2 or not 1.4<ratio<4:continue
        word=cv2.perspectiveTransform(uv[None],cv2.getPerspectiveTransform(unit,q))[0]
        extent=register_extent(profile,frame,word,polarity)
        if not extent['supported']:continue
        found.append(dict(word_quad=extent['word_quad'],carrier_quad=q.tolist(),
            print_identity_score=extent['score'],track_method='independent_carrier_border_flow',
            extent_refinement=dict(extent,proposal_quad=word.tolist())))
    found.sort(key=lambda x:x['print_identity_score'],reverse=True)
    unique=[]
    for candidate in found:
        q=np.float32(candidate['word_quad']);area=abs(cv2.contourArea(q))
        if any(cv2.intersectConvexConvex(q,np.float32(p['word_quad']))[0]/max(1,area)>.5 for p in unique):continue
        unique.append(candidate)
    return unique


def register_carrier(previous,current,carrier_quad,word_quad):
    carrier=np.float32(carrier_quad);word=np.float32(word_quad)
    mask=np.zeros(previous.shape,np.uint8)
    cv2.fillConvexPoly(mask,np.int32(carrier),255)
    ink_mask=np.zeros(previous.shape,np.uint8)
    cv2.fillConvexPoly(ink_mask,np.int32(word),255)
    mask[cv2.dilate(ink_mask,np.ones((31,31),np.uint8))!=0]=0
    points=cv2.goodFeaturesToTrack(previous,200,.003,3,mask=mask)
    if points is None or len(points)<8:return None
    # Coarse pyramid cells may include the changing print even though feature
    # centers exclude it. Retry finer scales with the same FB/RANSAC criteria.
    for level in (4,2,1):
        moved,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(31,31),maxLevel=level)
        back,reverse,_=cv2.calcOpticalFlowPyrLK(current,previous,moved,None,winSize=(31,31),maxLevel=level)
        good=(status.ravel()>0)&(reverse.ravel()>0)&(np.linalg.norm(points-back,axis=2).ravel()<1.)
        if good.sum()<8:continue
        matrix,inliers=cv2.estimateAffine2D(points[good],moved[good],method=cv2.RANSAC,ransacReprojThreshold=1.2)
        if matrix is None or inliers.sum()<8:continue
        scales=np.linalg.svd(matrix[:,:2],compute_uv=False)
        if not(.8<min(scales)<=max(scales)<1.25):continue
        return dict(carrier_quad=cv2.transform(carrier[None],matrix)[0].tolist(),
                    word_quad=cv2.transform(word[None],matrix)[0].tolist(),
                    registration_support=int(inliers.sum()),track_method='independent_carrier_border_flow')
    return None


def border_score(reference,current,root_carrier,root_word,carrier):
    plane=np.float32([[0,0],[319,0],[319,159],[0,159]])
    a=cv2.getPerspectiveTransform(np.float32(root_carrier),plane)
    b=cv2.getPerspectiveTransform(np.float32(carrier),plane)
    aa=cv2.warpPerspective(reference,a,(320,160));bb=cv2.warpPerspective(current,b,(320,160))
    valid=cv2.warpPerspective(np.full(current.shape,255,np.uint8),b,(320,160))>250
    mask=np.uint8(valid)*255
    q=cv2.perspectiveTransform(np.float32(root_word)[None],a)[0]
    cv2.fillConvexPoly(mask,np.int32(q),0)
    selected=cv2.erode(mask,np.ones((5,5),np.uint8))!=0
    if selected.sum()<300 or min(aa[selected].std(),bb[selected].std())<2:return -1.
    return float(np.corrcoef(aa[selected],bb[selected])[0,1])


def carrier_candidates(grays,references,scale=1,max_distance=None):
    candidates=[[]for _ in grays]
    for index,reference in enumerate(references):
        seed=reference['frame'];root=np.float32(reference['carrier_quad'])*scale;word=np.float32(reference['word_quad'])*scale
        def entry(n,found,score):
            return dict(found,word_quad=(np.float32(found['word_quad'])/scale).tolist(),
                        carrier_quad=(np.float32(found['carrier_quad'])/scale).tolist(),
                        state='visible',instance_id=f'border-{index}-{n}',anchor_frame=seed,
                        carrier_border_score=score,root_distance=abs(n-seed))
        candidates[seed].append(entry(seed,dict(carrier_quad=root,word_quad=word,track_method='independent_carrier_border_flow',registration_support=0),1.))
        for direction in (-1,1):
            previous=seed;c=root.copy();w=word.copy()
            for n in range(seed+direction,len(grays) if direction>0 else -1,direction):
                if max_distance is not None and abs(n-seed)>max_distance:break
                found=register_carrier(grays[previous],grays[n],c,w)
                if found is None:break
                score=border_score(grays[seed],grays[n],root,word,found['carrier_quad'])
                if score<.7:break
                candidates[n].append(entry(n,found,score));c=np.float32(found['carrier_quad']);w=np.float32(found['word_quad']);previous=n
    return candidates


def augment_records(item,records,style,spec):
    """Fill missed carriers through the same refinement entry, using bound evidence."""
    import json,subprocess
    import project
    path=project.project_path(spec['carrier_scan'])
    code=project.ROOT/'scripts/local_carrier_scan.swift'
    if not path.exists():
        binary=project.ROOT/'runs/.tools'/('carrier-scan-'+project.digest(code)[:16])
        if not binary.exists():
            binary.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['swiftc',str(code),'-o',str(binary)],check=True,capture_output=True)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as stream:
            stream.write(json.dumps(dict(source_sha256=item['source_sha256'],scanner_sha256=project.digest(code),on_device=True))+'\n');stream.flush()
            subprocess.run([str(binary),str(project.project_path(item['source'])),'12'],stdout=stream,check=True)
    rows=[json.loads(line)for line in path.read_text().splitlines()]
    if (rows[0]['source_sha256']!=item['source_sha256'] or rows[0].get('scanner_sha256')!=project.digest(code) or not rows[-1].get('complete') or
        rows[-1]['decoded_frames']!=len(records) or
        [r['frame']for r in rows[1:-1]]!=list(range(0,len(records),rows[-1]['stride']))):
        raise ValueError('Incomplete or unbound independent carrier scan')
    cap=cv2.VideoCapture(str(project.project_path(item['source'])));grays=[];references=[]
    by_frame={r['frame']:r for r in rows[1:-1]}
    present=[bool(r['instances'])for r in records]
    queries=set(by_frame)
    # Resample appearance/disappearance and tracking breaks densely. This is
    # driven by input records, never a case-specific list of repair frames.
    boundaries=[n for n in range(len(records)) if present[n] and
                (n==0 or n==len(records)-1 or not present[n-1] or not present[min(n+1,len(records)-1)])]
    for boundary in boundaries:
        queries.update(range(max(0,boundary-12),min(len(records),boundary+13)))
    for n in range(len(records)):
        ok,frame=cap.read()
        if not ok:raise ValueError('Independent carrier source decode ended early')
        grays.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        if n not in queries:continue
        proposals=by_frame.get(n,{}).get('rectangles',[])+dark_surface_proposals(frame)
        found=relocate_rectangles(frame,proposals,spec['carrier_reference_quad'],
            spec['reference_quad'],style['source_print_profile'],style['polarity'])
        references.extend(dict(c,frame=n)for c in found)
        if found and not present[n]:
            queries.update(range(n+1,min(len(records),n+13)))
    cap.release()
    alternatives=carrier_candidates(grays,references,max_distance=12)
    from print_registration import register_extent
    added=0;replaced=0
    cap=cv2.VideoCapture(str(project.project_path(item['source'])))
    for record,candidates in zip(records,alternatives):
        ok,frame=cap.read()
        if not ok:raise ValueError('Carrier selection source decode ended early')
        for candidate in sorted(candidates,key=lambda x:-x['carrier_border_score']):
            extent=register_extent(style['source_print_profile'],frame,candidate['word_quad'],style['polarity'])
            if not extent['supported']:continue
            candidate['extent_refinement']=dict(extent,proposal_quad=candidate['word_quad'])
            candidate['word_quad']=extent['word_quad']
            q=np.float32(candidate['word_quad']);area=abs(cv2.contourArea(q))
            overlaps=[i for i in record['instances'] if cv2.intersectConvexConvex(q,np.float32(i['word_quad']))[0]/max(1,area)>.35]
            if overlaps:
                scores=[register_extent(style['source_print_profile'],frame,i['word_quad'],style['polarity'])['score']for i in overlaps]
                if extent['score']<=max(scores)+.04:continue
                candidate['replaced_geometry']=overlaps
                record['instances']=[i for i in record['instances'] if i not in overlaps];replaced+=len(overlaps)
            else:added+=1
            record['instances'].append(candidate)
    cap.release()
    return dict(scan=spec['carrier_scan'],scan_sha256=project.digest(path),
                independent_roots=len(references),roots=references,queried_frames=sorted(queries),
                search_policy='uniform_and_record_boundaries_then_reacquisition',
                added_instances=added,replaced_instances=replaced)
