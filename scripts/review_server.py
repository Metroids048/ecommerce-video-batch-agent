"""Loopback-only review UI. Serves registered media; no arbitrary project files/commands."""
import json
import mimetypes
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import project
import workflow


def record_browser_event(case, body):
    event = {k: body.get(k) for k in ['event', 'file_id', 'candidate_id', 'source_time', 'duration', 'error']}
    if event['file_id'] not in {f['id'] for f in case['files']}:
        raise ValueError('Unknown playback file')
    event.update(batch_id=case['batch_id'], case_version=case['case_version'], recorded_at=workflow.now())
    evidence = workflow.state_path(case).with_name('browser-events.jsonl')
    evidence.parent.mkdir(parents=True, exist_ok=True)
    with evidence.open('a') as stream:
        stream.write(json.dumps(event, ensure_ascii=False)+'\n')
    return {'recorded': True}


def serve(case, case_relative, port):
    workflow.initialize(case, case_relative)
    from runtime_checkpoint import capture
    loaded_checkpoint=capture(case)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def json_response(self, value, code=200):
            data=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)

        def allowed(self):
            state=project.read_json(workflow.state_path(case));paths={case['user_input']}
            for item in case['files']:
                paths.add(item['source'])
                for op in item['operations']:
                    if op.get('replacement_asset'):paths.add(op['replacement_asset'])
                entry=state['files'][item['id']]
                paths.update(r['path'] for r in entry['raw_returns'])
                for c in entry['candidates']:
                    paths.add(c['output'])
                    paths.add(c['sidecar'])
                    paths.update(r['evidence'] for r in c.get('reviews',[]) if r.get('evidence'))
                    for operation in c.get('operation_reviews', []):
                        for check in operation['checks']:
                            paths.update(e['path'] for e in check.get('evidence', []))
            return paths

        def send_file(self, path, content_type=None):
            size=path.stat().st_size;start=0;end=size-1;partial=False
            header=self.headers.get('Range')
            if header:
                m=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
                if not m or int(m[1])>=size:
                    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
                start=int(m[1]);end=min(int(m[2]) if m[2] else end,end);partial=True
                if end<start:self.send_error(416);return
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type',content_type or mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
            self.send_header('Cache-Control','no-cache');self.send_header('X-Content-Type-Options','nosniff')
            if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with path.open('rb') as stream:
                    stream.seek(start);remaining=end-start+1
                    while remaining:
                        data=stream.read(min(256*1024,remaining))
                        if not data:break
                        self.wfile.write(data);remaining-=len(data)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            try:
                url=urlparse(self.path);q=parse_qs(url.query)
                if url.path in ('/','/review'):
                    return self.send_file(project.ROOT/'web/review.html','text/html; charset=utf-8')
                if url.path=='/api/runtime':return self.json_response(loaded_checkpoint)
                if url.path=='/api/batch':return self.json_response(workflow.snapshot(case))
                if url.path=='/api/delivery':return self.json_response(workflow.delivery_report(case))
                if url.path=='/api/plan':
                    state=project.read_json(workflow.state_path(case))
                    fid=q['file'][0];candidate=workflow.candidate_for(state['files'][fid],q['candidate'][0])
                    workflow.verify_candidate(case,fid,candidate)
                    plan=project.read_json(project.project_path(candidate['plan']))
                    if plan.get('schema') == 'execution_manifest_v1':
                        records=[]
                        for operation in plan['operations']:
                            if operation['disposition'] != 'ready':
                                continue
                            sub=project.read_json(project.project_path(operation['plan']))
                            records.extend(dict(r, source_frame=r['frame'], operation_id=operation['operation_id'])
                                           for r in sub['review'].get('records', []))
                        return self.json_response({'source_frames_half_open':[0,state['files'][fid]['source_frame_count']],
                            'records':records,'frame_states':None,'operations':candidate.get('operation_results',[]),
                            'supervised':True,'sampling_mode':'unified_source_methods','spec':{}})
                    records=[dict(r,source_frame=plan['source_frames_half_open'][0]+r['frame'])
                             for r in plan['review'].get('records',[])]
                    # Relevant supervised geometry, not full NPZ arrays or model logs.
                    return self.json_response({'frame_states':plan['review'].get('frame_states'),
                        'source_frames_half_open':plan['source_frames_half_open'],'records':records,
                        'supervised':True,'sampling_mode':plan.get('sampling_mode'),'spec':{
                            k:plan.get('spec',{}).get(k) for k in ['cloth_polygons','fully_occluded_intervals']}})
                if url.path=='/media':
                    value=q.get('path',[''])[0]
                    if value not in self.allowed():raise ValueError('Unregistered media/evidence path')
                    return self.send_file(project.project_path(value))
                self.send_error(404)
            except (ValueError,KeyError,OSError) as exc:self.json_response({'error':str(exc)},400)

        def do_POST(self):
            try:
                expected=f'http://127.0.0.1:{port}'
                if self.headers.get('Host')!=f'127.0.0.1:{port}' or self.headers.get('Origin')!=expected:
                    return self.json_response({'error':'Only same-origin loopback UI may mutate this project'},403)
                length=int(self.headers.get('Content-Length','0'))
                if length<=0 or length>16384 or self.headers.get('Content-Type')!='application/json':raise ValueError('Expected bounded JSON body')
                body=json.loads(self.rfile.read(length));path=urlparse(self.path).path
                if path=='/api/issues':return self.json_response(workflow.add_feedback(case,body),201)
                if path=='/api/repair-plan':return self.json_response(workflow.plan_feedback(case,body))
                if path=='/api/repairs':return self.json_response(workflow.begin_repair(case,body),202)
                if path=='/api/events':return self.json_response(record_browser_event(case, body))
                self.send_error(404)
            except ValueError as exc:self.json_response({'error':str(exc)},409 if 'conflict' in str(exc).lower() else 400)
            except (KeyError,OSError,TypeError) as exc:self.json_response({'error':str(exc)},400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    workflow.reconcile_interrupted(case)
    print(json.dumps({'review_url':f'http://127.0.0.1:{port}/','case':case_relative,
                      'pid':__import__('os').getpid(),'paid_submission_allowed':False},ensure_ascii=False),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
    return 0
