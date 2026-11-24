"""Copy the current verified workflow into a portable, read-only review handoff."""
import html
import json
import shutil
from datetime import datetime, timezone

import project
import workflow


def build(case, out, diagnostic=False, full_comparison=False):
    from production_pipeline import controlled
    if controlled(case) and case['execution_policy']['mode'] == 'method_trial' and not diagnostic:
        raise ValueError('A method research result cannot become a normal production delivery')
    report = workflow.delivery_report(case)  # verifies live source/candidate dependencies
    state = project.read_json(workflow.state_path(case))
    if state['revision'] != report['revision']:
        raise ValueError('Workflow changed while preparing delivery; retry from current state')
    for entry in report['files']:
        if entry['candidate']:
            workflow.verify_candidate(case, entry['id'], entry['candidate'])
    if any(job['status'] in ('queued', 'running') for entry in state['files'].values() for job in entry['jobs']):
        raise ValueError('Wait for existing local jobs before freezing delivery')
    unqualified = [entry.get('id', 'unknown') for entry in report['files']
                   if entry.get('delivery_status') != 'qualified' or entry.get('unresolved_issues')]
    if unqualified and not diagnostic:
        raise ValueError('Unqualified videos cannot be delivered: ' + ', '.join(unqualified)
                         + '; use --diagnostic only for explicitly labelled failure evidence')
    root = project.project_path(out)
    if root.exists():
        metadata = project.read_json(root / 'package.json')
        previous = project.read_json(root / 'delivery-report.json')
        binding = lambda r: [(f['id'], f['current_candidate_id'],
                               f['candidate']['output_sha256'] if f['candidate'] else None,
                               f.get('delivery_status'), f.get('unresolved_issues')) for f in r['files']]
        if (metadata['batch_id'] != case['batch_id'] or binding(previous) != binding(report)
                or metadata['status'] != ('failure_diagnostic' if diagnostic else 'qualified_delivery')):
            raise ValueError('Existing delivery belongs to different results; preserve it')
        comparisons = [c for c in metadata['copied_files'] if c['path'].endswith('-comparison.mp4')]
        if full_comparison and len(comparisons) != sum(bool(f['candidate']) for f in report['files']):
            raise ValueError('Existing delivery lacks requested comparison; preserve it')
        for entry in metadata['copied_files']:
            artifact = (root / entry['path']).resolve()
            if not artifact.is_relative_to(root.resolve()) or project.digest(artifact) != entry['sha256']:
                raise ValueError('Existing delivery artifact changed')
        return dict(status=metadata['status'], output=out, execution_reused=True,
                    new_model_calls=0, new_renders=0, accepted=False)
    root.mkdir(parents=True, exist_ok=False)  # never replace an earlier handoff
    copied = []

    def copy(relative, name, expected=None):
        source = project.project_path(relative)
        original = project.digest(source)
        if expected is not None and original != expected:
            raise ValueError('Delivery input drift: ' + relative)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if project.digest(target) != original:
            raise ValueError('Delivery copy mismatch: ' + name)
        copied.append({'path': name, 'sha256': original, 'bytes': target.stat().st_size,
                       'project_source': relative})
        return name

    esc = html.escape
    user_input = copy(case['user_input'], '用户输入.txt')
    sections, notes = [], []
    provenance = case.get('provenance', '未提供，不推断客户委托、盲测或发布授权。')
    if not isinstance(provenance, str):
        provenance = json.dumps(provenance, ensure_ascii=False)
    notes.extend([('失败诊断材料，不是可用成片；原目标尚未完成。' if diagnostic else
                   '本批逐项质量检查通过，用户是否接受另行记录。'),
                  '素材及订单来源：' + provenance,
                  '前后图是实际源片与本版编码输出的抽样，不能代替全片连续视听。',
                  '技术检查、Agent观察和原生播放分别记录在详细清单中；逐包核对与顺序解码不等于实际听审或连续视觉观看。',
                  '远程操作按本批授权及预算执行；本地回填／返修与模型原始效果分开，漏改不能归因于接口成功或程序检查。'])
    if case.get('execution_policy', {}).get('reuse_only'):
        notes.append('本批仅复用同源、同结构化改单的完整历史结果并重新审查；未新增模型调用、上传或成片制作。历史方法、任务及费用见各片缓存来源；本轮不证明新素材生成能力或速度。')
    for ordinal, entry in enumerate(report['files'], 1):
        item = workflow.item_for(case, entry['id'])
        candidate = entry['candidate']
        source = copy(item['source'], f'media/{ordinal}-original' + project.project_path(item['source']).suffix,
                      item['source_sha256'])
        body = [f'<h2>{esc(item.get("title", entry["id"]))}</h2>',
                f'<p>状态：{esc(entry["delivery_status"])}。版本：{esc(str(entry["current_candidate_id"]))}</p>']
        notes.append(f'\n{entry["id"]}：{entry["delivery_status"]}，版本 {entry["current_candidate_id"]}')
        if candidate:
            output = copy(candidate['output'], f'media/{ordinal}-edited.mp4', candidate['output_sha256'])
            if candidate.get('cache_origin'):
                receipt = root/f'evidence/{ordinal}-cache-origin.json'
                receipt.parent.mkdir(parents=True, exist_ok=True)
                receipt.write_text(json.dumps(candidate['cache_origin'], ensure_ascii=False, indent=2)+'\n')
                copied.append(dict(path=str(receipt.relative_to(root)), sha256=project.digest(receipt),
                    bytes=receipt.stat().st_size, project_source='derived exact-cache provenance'))
                notes.append('缓存来源：'+json.dumps(candidate['cache_origin'], ensure_ascii=False))
            if full_comparison:
                import subprocess
                width,height=item['dimensions'];tile_width=360 if height>width else 640
                tile_height=2*round(height*tile_width/width/2)
                comparison=root/f'media/{ordinal}-comparison.mp4'
                scale=f'scale={tile_width}:{tile_height},setsar=1'
                subprocess.run(['ffmpeg','-v','error','-i',str(root/source),'-i',str(root/output),
                    '-filter_complex',f'[0:v]{scale}[a];[1:v]{scale}[b];[a][b]hstack=inputs=2[v]',
                    '-map','[v]','-map','0:a:0?','-c:v','libx264','-preset','fast','-crf','19',
                    '-c:a','copy','-movflags','+faststart',str(comparison)],check=True)
                copied.append(dict(path=str(comparison.relative_to(root)),sha256=project.digest(comparison),
                    bytes=comparison.stat().st_size,project_source='derived full-length source/output comparison'))
            copy(candidate['sidecar'], f'evidence/{ordinal}-technical.json')
            body.append(f'<div class="pair"><figure><figcaption>完整原片</figcaption><video controls preload="metadata" src="{esc(source)}"></video><a download href="{esc(source)}">下载原片</a></figure>'
                        f'<figure><figcaption>{"失败诊断候选 · 不可作为成片使用" if diagnostic else "完整修改视频"}</figcaption><video controls preload="metadata" src="{esc(output)}"></video><a download href="{esc(output)}">{"下载失败诊断候选" if diagnostic else "下载修改视频"}</a></figure></div>')
            if full_comparison:
                body.append(f'<p><a href="media/{ordinal}-comparison.mp4">播放完整前后对比</a> · <a download href="media/{ordinal}-comparison.mp4">下载完整前后对比</a></p>')
            previews = []
            for path, expected in candidate.get('artifact_hashes', {}).items():
                if not path.endswith('/queue.json'):
                    continue
                if project.digest(project.project_path(path)) != expected:
                    raise ValueError('Review queue drift: ' + path)
                queue = project.read_json(project.project_path(path))
                if (queue.get('output_sha256') != candidate['output_sha256']
                        or queue.get('source_sha256') != item['source_sha256']):
                    continue
                for sheet in queue.get('overview_sheets', []):
                    preview = copy(sheet, f'previews/{ordinal}-{len(previews)+1}.jpg')
                    previews.append(preview)
            for preview in previews:
                body.append(f'<details><summary>前后对照 · 左原片／右本版</summary><a href="{preview}"><img loading="lazy" src="{preview}" alt="实际源片与本版输出抽样对照"></a></details>')
            body.append('<h3>逐项检查</h3><ul>')
            for operation in item['operations']:
                observations = next((r for r in candidate.get('operation_reviews', [])
                                     if r['operation_id'] == operation['id']), {})
                checks = observations.get('checks', [])
                failed = [check['scope'] for check in checks if check['status'] != 'pass']
                text = ('；'.join(failed) if failed else
                        ('逐项检查通过；用户是否接受另行记录。' if checks else '逐项检查未提供，不能据此认为完成'))
                body.append(f'<li><b>{esc(operation["target"])}</b><p>{esc(text)}</p></li>')
                notes.append(f'- {operation["target"]}：{text}')
            body.append('</ul>')
        else:
            body.append(f'<p>尚无完整修改版：{esc(str(entry.get("missing_reason")))}</p>')
        body.append('<h3>仍需处理</h3><ul>')
        for issue in entry['unresolved_issues']:
            version = issue.get('candidate_id', 'unknown')
            context = ('本版问题' if version == entry['current_candidate_id'] else '父版问题，后续返修尚未整项关闭；不代表原缺陷仍全部存在')
            text = f'[{context} · {version}] {issue.get("source_time_seconds", 0):.3f}秒／源帧{issue.get("source_frame")}：{issue["text"]}'
            body.append(f'<li>{esc(text)}</li>')
            notes.append('- ' + text)
        body.append('</ul><p>全部修改区间、未解决范围、父版和检查依赖见 <a href="delivery-report.json">详细清单</a>。区间使用源帧半开范围；未知帧不等于目标在场，也不能当作无需修改。</p>')
        sections.append('\n'.join(body))

    # A handoff is a snapshot, never a second writable workflow or acceptance record.
    current = project.read_json(workflow.state_path(case))
    if current['revision'] != report['revision']:
        raise ValueError('Workflow changed during delivery; this folder is incomplete')
    created = datetime.now(timezone.utc).isoformat()
    jobs = [dict(job) for entry in state['files'].values() for job in entry['jobs']]
    elapsed = (datetime.fromisoformat(created) - datetime.fromisoformat(state['created_at'])).total_seconds()
    metadata = {'batch_id': case['batch_id'], 'workflow_revision': report['revision'],
                'created_at': created, 'status': 'failure_diagnostic' if diagnostic else 'qualified_delivery', 'accepted': False,
                'original_goal_complete': report['original_goal_complete'],
                'case_provenance': case.get('provenance'), 'copied_files': copied,
                'jobs_including_failed_parents': jobs,
                'workflow_initialized_at': state['created_at'],
                'workflow_initialized_to_package_wall_seconds': elapsed,
                'time_scope': 'Starts at review-init, excluding earlier input intake/preparation. Includes overlapping development, Agent observation and local renders; not user-only latency or measured labor savings.',
                'packaging_only_new_remote_calls': 0, 'packaging_only_new_cost_cny': 0,
                'batch_cost': workflow.cost_summary(state)}
    timing_path=project.ROOT/'runs'/case['batch_id']/'timings.jsonl'
    if timing_path.is_file():
        copy(str(timing_path.relative_to(project.ROOT)), 'evidence/command-timings.jsonl')
    stage_path=timing_path.with_name('stage-events.jsonl')
    if stage_path.is_file():
        copy(str(stage_path.relative_to(project.ROOT)), 'evidence/stage-events.jsonl')
    metadata['timing_summary']=project.timing_summary(case,until=created)
    (root / 'delivery-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (root / '交付说明.md').write_text('\n\n'.join(notes)+'\n')
    title = '批量修整 · 失败诊断材料' if diagnostic else '批量修整 · 修改视频交付'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>批量修整 · 完整审查版</title><style>body{font:16px/1.7 system-ui;margin:32px auto;padding:0 24px;max-width:1100px;background:#f7f6f2;color:#242823}h1{margin-bottom:8px}.notice{padding:18px;background:#fff1d9;border-left:4px solid #a65e15}.pair{display:flex;gap:20px}figure{margin:0;flex:1;min-width:0}video{width:100%;max-height:560px;background:#111}section{background:white;padding:24px;margin:24px 0;border:1px solid #ddd}img{max-width:100%;height:auto}a{color:#266446}summary{cursor:pointer;padding:12px}li p{margin-top:4px}@media(max-width:650px){.pair{display:block}figure{margin-bottom:24px}body{padding:0 12px}}</style><h1>批量修整 · 完整审查版</h1>'
    page = page.replace('批量修整 · 完整审查版', title)
    page += '<div class="notice">' + ''.join('<p>'+esc(n)+'</p>' for n in notes[:5]) + '</div>'
    page += f'<p>订单：{esc(case["batch_id"])} · <a href="{user_input}">原始修改要求</a> · <a href="交付说明.md">未解决项</a> · <a href="package.json">文件校验与实际执行记录</a></p>'
    page += ''.join('<section>'+section+'</section>' for section in sections)+'</html>'
    (root / 'index.html').write_text(page)
    (root / 'package.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    return {'status': metadata['status'], 'package': out, 'revision': report['revision'],
            'file_count': len(report['files']), 'original_goal_complete': False, 'accepted': False}
