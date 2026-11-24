"""Source-bound observation material. Images never count as Agent observations."""
import json
import cv2
import numpy as np

import project


def build(case, file_id, out, plan=None, frames=None, stride=30, candidate_id=None):
    item = next(f for f in case['files'] if f['id'] == file_id)
    source = project.project_path(item['source'])
    if project.digest(source) != item['source_sha256']:
        raise ValueError('Observation source changed')
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('Invalid observation stride')
    destination = project.project_path(out)
    if destination.exists():
        raise ValueError('Preserve the previous observation queue')
    cap = cv2.VideoCapture(str(source))
    candidate = None
    result_cap = None
    if candidate_id:
        import workflow
        state = project.read_json(workflow.state_path(case))
        candidate = workflow.verify_candidate(case, file_id,
                    workflow.candidate_for(state['files'][file_id], candidate_id))
        result_cap = cv2.VideoCapture(str(project.project_path(candidate['output'])))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = set(frames if frames is not None else range(0, count, stride))
    selected.update([0, count-1])
    pending = []
    if plan is not None:
        if plan['file_id'] != file_id or plan['source_sha256'] != item['source_sha256']:
            cap.release()
            raise ValueError('Observation plan belongs to another source')
        states = plan['review']['frame_states']
        if len(states) != count:
            cap.release()
            raise ValueError('Observation plan frame count changed')
        lo = None
        for n, state in enumerate(states + ['end']):
            if state == 'unresolved' and lo is None:
                lo = n
            elif state != 'unresolved' and lo is not None:
                pending.append([lo, n])
                for boundary in (lo, (lo+n-1)//2, n-1):
                    selected.update(range(max(0, boundary-2), min(count, boundary+3)))
                lo = None
    if any(not isinstance(n, int) or n < 0 or n >= count for n in selected):
        cap.release()
        raise ValueError('Observation frame outside source')
    destination.mkdir(parents=True)
    images, cells = [], []
    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if result_cap is not None:
                result_ok, result = result_cap.read()
                if not result_ok or result.shape != frame.shape:
                    raise ValueError('Candidate cannot be paired with complete source')
            if decoded in selected:
                height, width = frame.shape[:2]
                material = np.concatenate([frame, result], axis=1) if candidate else frame
                cell_width = 480 if candidate else 240
                thumbnail = cv2.resize(material, (cell_width, max(1, round(height*240/width))))
                cv2.putText(thumbnail, 'source frame '+str(decoded), (4, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                filename = destination/f'frame-{decoded:06d}.jpg'
                if not cv2.imwrite(str(filename), material):
                    raise ValueError('Cannot write observation frame')
                images.append({'source_frame': decoded, 'path': str(filename.relative_to(project.ROOT)),
                               'sha256': project.digest(filename), 'observation_status': 'pending'})
                cells.append(thumbnail)
            decoded += 1
    finally:
        cap.release()
        if result_cap is not None:
            extra, _ = result_cap.read()
            result_cap.release()
            if extra:
                raise ValueError('Candidate has extra frames')
    if decoded != count:
        raise ValueError('Observation decode count differs; partial material preserved')
    sheets = []
    for offset in range(0, len(cells), 24):
        page = cells[offset:offset+24]
        h = max(c.shape[0] for c in page)
        cell_width = 480 if candidate else 240
        canvas = np.zeros((h*((len(page)+3)//4), cell_width*4, 3), np.uint8)
        for i, cell in enumerate(page):
            canvas[(i//4)*h:(i//4)*h+cell.shape[0], (i%4)*cell_width:(i%4+1)*cell_width] = cell
        path = destination/f'overview-{offset//24+1:03d}.jpg'
        cv2.imwrite(str(path), canvas)
        sheets.append(str(path.relative_to(project.ROOT)))
    manifest = {'schema_version': 1, 'file_id': file_id, 'case_version': case['case_version'],
                'source_sha256': item['source_sha256'], 'source_frame_count': decoded,
                'operation_id': plan.get('operation_id') if plan else None,
                'unresolved_source_frame_ranges': pending, 'images': images, 'overview_sheets': sheets,
                'status': 'pending_agent_observation', 'full_visual_reviewed': False,
                'candidate_id': candidate_id, 'output_sha256': candidate['output_sha256'] if candidate else None,
                'layout': 'source left, candidate right' if candidate else 'source only',
                'scope': 'Full source decoded; sampled overview and unresolved boundary sequences only. Not semantic coverage or continuous viewing.'}
    (destination/'queue.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest


def pair_inputs(case, file_id, spec, candidate_id=None):
    import workflow
    item=workflow.item_for(case,file_id)
    source=project.project_path(item['source'])
    if project.digest(source)!=item['source_sha256']:raise ValueError('Observation source changed')
    if spec.get('source_only') is True:
        return item,source,source,0,item['source_sha256']
    state=project.read_json(workflow.state_path(case))
    if spec.get('maintenance_probe'):
        probe=state['files'][file_id]['component_probe'];raw=probe['output']
        if probe['recipe']['dependency'] != workflow.requirement_binding(case,item):
            raise ValueError('Maintenance probe source/requirements changed')
        path=project.project_path(raw['path'])
        if project.digest(path)!=raw['sha256']:raise ValueError('Maintenance probe output changed')
        return item,source,path,raw['source_frames_half_open'][0],raw['sha256']
    if spec.get('trial_id'):
        trial=next(t for t in state['files'][file_id].get('preview_trials',[]) if t['id']==spec['trial_id'])
        report_path=project.project_path(trial['report'])
        if project.digest(report_path)!=trial['report_sha256']:raise ValueError('Preview report changed')
        report=project.read_json(report_path);path=project.project_path(report['output'])
        if report['source_sha256']!=item['source_sha256'] or project.digest(path)!=report['output_sha256']:
            raise ValueError('Preview media binding changed')
        return item,source,path,report['source_frames_half_open'][0],report['output_sha256']
    if candidate_id:
        candidate=workflow.verify_candidate(case,file_id,workflow.candidate_for(state['files'][file_id],candidate_id))
        return item,source,project.project_path(candidate['output']),0,candidate['output_sha256']
    from runninghub_adapter import unit_for, verify_unit
    unit=unit_for(state,file_id,spec['unit_id']);request=verify_unit(case,unit)
    raw=unit['raw'];path=project.project_path(raw['path'])
    if project.digest(path)!=raw['sha256']:raise ValueError('Observation raw changed')
    return item,source,path,request['source_frames_half_open'][0],raw['sha256']


def build_pairs(case,file_id,out,spec,candidate_id=None):
    """All decoded pairs in compact sheets; no quality verdict or individual-frame dumps."""
    from PIL import Image,ImageDraw
    item,source,result,start,expected=pair_inputs(case,file_id,spec,candidate_id)
    destination=project.project_path(out)
    destination.mkdir(parents=True,exist_ok=False)
    source_only=spec.get('source_only') is True
    a=cv2.VideoCapture(str(source));b=None if source_only else cv2.VideoCapture(str(result));a.set(1,start)
    count=int((a if source_only else b).get(7));stills=set(spec.get('stills',[]));evidence=[];frames=[];evidence_frames={}
    if count < 1:
        a.release()
        if b is not None:b.release()
        raise ValueError('Observation has no decoded frames')
    # The same fixed cell area supports both portrait and landscape, without
    # hundreds of full-size JPG writes on every review.
    sw,sh=int(a.get(3)),int(a.get(4));tw,th=(144,256) if sh>sw else (256,144)
    rows=6;cols=6;page=None;cell_width=tw if source_only else tw*2
    try:
        for i in range(count):
            x,src=a.read();y,dst=(x,src) if source_only else b.read()
            if not x or not y:raise ValueError('Incomplete source/result pairing')
            if i%36==0:
                page=Image.new('RGB',(cols*cell_width,rows*(th+23)),'#171717');draw=ImageDraw.Draw(page)
            for j,im in enumerate((src,) if source_only else (src,dst)):
                tile=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));tile.thumbnail((tw,th))
                page.paste(tile,((i%cols)*cell_width+j*tw,((i%36)//cols)*(th+23)+21))
            draw.text(((i%cols)*cell_width+3,((i%36)//cols)*(th+23)+3),f'{start+i} source' if source_only else f'{start+i} source | edit',fill='white')
            if start+i in stills:
                h=min(src.shape[0],dst.shape[0])
                pair=np.concatenate([cv2.resize(im,(round(im.shape[1]*h/im.shape[0]),h)) for im in ((src,) if source_only else (src,dst))],axis=1)
                path=destination/f'native-{start+i:04d}.jpg';cv2.imwrite(str(path),pair);evidence.append(path);evidence_frames[path]=[start+i]
            frames.append(start+i)
            if i%36==35 or i==count-1:
                path=destination/f'all-{i//36:03d}.jpg';page.save(path,quality=93);evidence.append(path);evidence_frames[path]=list(range(start+(i//36)*36,start+i+1))
        extra,_=(a if source_only else b).read()
        if extra:raise ValueError('Extra result frames')
    finally:
        a.release()
        if b is not None:b.release()
    if project.digest(result)!=expected:raise ValueError('Result changed during observation')
    inventory=dict(file_id=file_id,candidate_id=candidate_id,source_sha256=item['source_sha256'],
        output_sha256=expected,result=str(result.relative_to(project.ROOT)),source_frames_half_open=[start,start+count],
        paired_frames=frames,source_only=source_only,dimensions=[sw,sh],
        evidence=[dict(path=str(path.relative_to(project.ROOT)),sha256=project.digest(path),frames=evidence_frames[path],
                       native_size=path.name.startswith('native-')) for path in evidence],
        overview_sheets=[str(path.relative_to(project.ROOT)) for path in evidence if path.name.startswith('all-')],
        status='pending_actual_agent_observation',scope='Every result frame paired; sheets and selected native-size pairs require actual Agent inspection. No automatic visual pass.')
    (destination/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    return dict(status=inventory['status'],frame_pairs=count,sheets=len(inventory['overview_sheets']),inventory=str((destination/'inventory.json').relative_to(project.ROOT)))


def native_playback(case,file_id,out,candidate_id=None,spec=None):
    """Play the verified candidate in Chrome at 1x, capture evidence, then stop owned processes."""
    import functools,http.server,subprocess,tempfile,threading,time,shutil
    from pathlib import Path
    if not candidate_id and not (spec or {}).get('maintenance_probe'):raise ValueError('Native playback requires candidate')
    _,_,result,_,expected=pair_inputs(case,file_id,spec or {},candidate_id)
    destination=project.project_path(out);destination.mkdir(parents=True,exist_ok=False)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
        def copyfile(self,source,outputfile):
            try:super().copyfile(source,outputfile)
            except (BrokenPipeError,ConnectionResetError):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(project.ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    node=shutil.which('node')
    if not node:raise ValueError('Node.js is required for native playback')
    with tempfile.TemporaryDirectory(prefix='aiw-native-review-') as profile:
        process=subprocess.Popen([chrome,'--headless=new','--no-first-run','--no-default-browser-check',
            '--remote-debugging-port=0','--autoplay-policy=no-user-gesture-required','--user-data-dir='+profile,'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            portfile=Path(profile)/'DevToolsActivePort'
            for _ in range(100):
                if portfile.is_file():break
                time.sleep(.1)
            port=portfile.read_text().splitlines()[0]
            url=f'http://127.0.0.1:{server.server_port}/'+str(result.relative_to(project.ROOT))
            completed=subprocess.run([node,str(project.ROOT/'scripts/native_playback.mjs'),url,str(destination/'play'),port],text=True,capture_output=True,timeout=180)
            if completed.returncode:raise ValueError('Native playback failed: '+completed.stderr[-500:])
            if project.digest(result)!=expected:raise ValueError('Candidate changed during playback')
            return json.loads(completed.stdout)
        finally:
            process.terminate();process.wait(timeout=10);server.shutdown();server.server_close()
