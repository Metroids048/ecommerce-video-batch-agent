"""Source observation handoff, not a semantic detector or an automatic verdict.

The host inspects actual images and records every operation/frame/instance.
Receipts preserve partial work and uncertainty; consumers check the same object.
"""
import json
import math

import project
import workflow


def required(case):
    policy=case.get('execution_policy', {})
    return policy.get('source_semantic_review_required') is True and policy.get('reuse_only') is not True


def artifact(path):
    return dict(path=str(path.relative_to(project.ROOT)),sha256=project.digest(path))


def read_artifact(ref, label):
    path=project.project_path(ref['path'])
    if project.digest(path)!=ref['sha256']:
        raise ValueError(label+' changed')
    return project.read_json(path)


def queue_for(case, fid, saved):
    item=workflow.item_for(case,fid)
    queue=read_artifact(saved['queue'],'Source observation queue')
    count=item['source_frame_count_inventory']
    if (queue.get('dependency')!=workflow.requirement_binding(case,item)
            or queue.get('file_id')!=fid or queue.get('source_only') is not True
            or queue.get('source_sha256')!=item['source_sha256']
            or queue.get('operations')!=item['operations']
            or queue.get('user_input')!=dict(path=case['user_input'],sha256=project.digest(project.project_path(case['user_input'])))
            or queue.get('paired_frames')!=list(range(count))
            or queue.get('dimensions')!=item['dimensions']
            or project.digest(project.project_path(item['source']))!=item['source_sha256']):
        raise ValueError('Source observation queue must bind the full source and current requirements')
    covered=set()
    for ref in saved.get('detail_queues',[]):
        detail=read_artifact(ref,'Source observation detail queue')
        if (detail.get('file_id')!=fid or detail.get('source_sha256')!=item['source_sha256']
                or detail.get('case_version')!=case['case_version']
                or detail.get('source_frame_count')!=count or detail.get('candidate_id') is not None
                or detail.get('layout')!='source only'):
            raise ValueError('Source observation details must bind this complete original source')
        for image in detail['images']:
            n=image['source_frame']
            if type(n) is not int or not 0<=n<count:
                raise ValueError('Source observation detail frame outside source')
            queue['evidence'].append(dict(path=image['path'],sha256=image['sha256'],frames=[n],native_size=True))
    for e in queue['evidence']:
        if project.digest(project.project_path(e['path']))!=e['sha256']:
            raise ValueError('Source observation evidence changed')
        covered.update(e['frames'])
    if covered!=set(range(count)):
        raise ValueError('Source observation images must cover every frame')
    return queue


def validate_rows(rows, queue):
    evidence={e['path']:e for e in queue['evidence']}
    seen=set();width,height=queue['dimensions'];count=len(queue['paired_frames'])
    for row in rows:
        n=row.get('frame');state=row.get('state');instances=row.get('instances')
        if type(n) is not int or not 0<=n<count or n in seen:
            raise ValueError('Source observation has duplicate or invalid frame')
        seen.add(n)
        if state not in ('visible','occluded','absent','unresolved'):
            raise ValueError('Source observation needs an explicit visibility state')
        if not isinstance(row.get('reason'),str) or not row['reason'].strip():
            raise ValueError('Source observation needs an actual observation reason, including absence/occlusion')
        if state!='unresolved' and row.get('all_instances_observed') is not True:
            raise ValueError('Source observation must account for all instances, or remain unresolved')
        if not isinstance(instances,list) or (state=='visible')!=bool(instances):
            raise ValueError('Visible source observation needs all target instances; other states keep no visible instances')
        ids=set()
        for instance in instances:
            ident=instance.get('id');box=instance.get('bbox')
            if not isinstance(ident,str) or not ident.strip() or ident in ids:
                raise ValueError('Source observation needs distinct instance IDs within each frame')
            ids.add(ident)
            if (not isinstance(box,list) or len(box)!=4
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in box)
                    or not (0<=box[0]<box[0]+box[2]<=width and 0<=box[1]<box[1]+box[3]<=height)):
                raise ValueError('Source observation instance bbox must use original source pixels')
        refs=row.get('evidence')
        if not isinstance(refs,list) or not refs:
            raise ValueError('Source observation requires inspected image evidence for each frame')
        for ref in refs:
            image=evidence.get(ref.get('path'))
            if not image or ref.get('sha256')!=image['sha256'] or n not in image['frames']:
                raise ValueError('Source observation evidence does not bind this frame')
        risks=row.get('risks',[])
        if not isinstance(risks,list) or not all(isinstance(r,str) and r.strip() for r in risks):
            raise ValueError('Source observation risks must be explicit reasons')


def summarize(case, operations):
    count=case['source_frame_count_inventory']
    summaries={};complete=True
    for op in case['operations']:
        data=operations.get(op['id'],{});rows=data.get('records',[])
        indices={r['frame'] for r in rows}
        missing=workflow.ranges([n for n in range(count) if n not in indices])
        unresolved=workflow.ranges([r['frame'] for r in rows if r['state']=='unresolved'])
        appearance=workflow.ranges([r['frame'] for r in rows if r['state'] in ('visible','occluded')])
        risk=workflow.ranges([r['frame'] for r in rows if r['state'] in ('occluded','unresolved')
                              or r.get('risks') or len(r['instances'])>1])
        by_frame={r['frame']:r for r in rows}
        transitions=[n for n in sorted(by_frame) if n>0 and n-1 in by_frame
                     and by_frame[n]['state']!=by_frame[n-1]['state']]
        risk=workflow.ranges(sorted({n for lo,hi in risk for n in range(lo,hi)} |
                                   {f for n in transitions for f in (n-1,n)}))
        complete=complete and not missing and not unresolved
        summaries[op['id']]=dict(missing_frames_half_open=missing,unresolved_frames_half_open=unresolved,
            appearance_ranges_half_open=appearance,risk_ranges_half_open=risk,
            quality_requirements=data.get('quality_requirements',[]))
    return summaries,complete


def verify(case, fid, entry, complete=False):
    saved=entry.get('source_observation')
    if not saved:
        raise ValueError('Prepare full source observation before freezing the route')
    queue=queue_for(case,fid,saved);item=workflow.item_for(case,fid)
    receipt=read_artifact(saved['receipt'],'Source observation receipt') if saved.get('receipt') else None
    if receipt:
        if receipt.get('semantic_execution'):
            from semantic_scan import read_result
            if receipt['semantic_execution']!=entry.get('source_semantic_execution',{}).get('result'):
                raise ValueError('Source review must preserve the exact semantic execution result')
            read_result(case,fid,entry)
        if (receipt.get('schema')!='source_observation_v1' or receipt.get('queue')!=saved['queue']
                or receipt.get('dependency')!=workflow.requirement_binding(case,item)
                or receipt.get('file_id')!=fid or receipt.get('source_sha256')!=item['source_sha256']
                or set(receipt.get('operations',{}))!={o['id'] for o in item['operations']}):
            raise ValueError('Source observation receipt has stale source/requirements/queue')
        for op in receipt['operations'].values():
            validate_rows(op['records'],queue)
            if op['records'] and (not op.get('quality_requirements') or
                    not all(isinstance(q,str) and q.strip() for q in op['quality_requirements'])):
                raise ValueError('Source observation needs operation quality requirements')
    summaries,done=summarize(item,receipt['operations'] if receipt else {})
    if receipt:
        for oid,summary in summaries.items():
            if any(receipt['operations'][oid].get(k)!=v for k,v in summary.items()):
                raise ValueError('Source observation summary differs from actual frame records')
    if complete and (not done or not receipt):
        raise ValueError('Full source observation incomplete; consume missing/unresolved frames before freeze-route')
    if complete and any(not s['appearance_ranges_half_open'] for s in summaries.values()):
        raise ValueError('Source observation found no target; preserve absence and clarify the operation')
    return receipt,summaries,done


def progress(case, fid, entry):
    saved=entry.get('source_observation')
    if not saved:
        return dict(next_action='prepare-source-observation',status='not_prepared')
    receipt,summaries,done=verify(case,fid,entry)
    found=all(s['appearance_ranges_half_open'] for s in summaries.values())
    result = dict(status='complete_host_observation' if done else 'pending_actual_agent_observation',
        next_action=('freeze-route' if found else 'clarify_absent_target') if done else 'record-source-observation',
        queue=saved['queue'],receipt=saved.get('receipt'),
        detail_queues=saved.get('detail_queues',[]),
        missing_frames_half_open={k:v['missing_frames_half_open'] for k,v in summaries.items()},
        unresolved_frames_half_open={k:v['unresolved_frames_half_open'] for k,v in summaries.items()},
        appearance_ranges_half_open={k:v['appearance_ranges_half_open'] for k,v in summaries.items()},
        scope='Host-declared image observations; structural completeness is checked, semantic truth and output quality are not automatically verified.')
    if not done and (entry.get('source_semantic_execution') or
                     case.get('execution_policy',{}).get('source_semantic_execution')=='mlx_vlm'):
        from semantic_scan import progress as scan_progress
        scan = scan_progress(case,fid,entry)
        result.update(semantic_execution=scan,next_action=scan['next_action'])
    return result


def prepare(case, spec):
    from production_pipeline import require_production
    from review_queue import build_pairs
    require_production(case)
    fid=spec['file_id'];item=workflow.item_for(case,fid)
    stills=spec.get('stills',[])
    if not isinstance(stills,list) or any(type(n) is not int or not 0<=n<item['source_frame_count_inventory'] for n in stills):
        raise ValueError('Source observation detail frames outside source')
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case,state);entry=state['files'][fid]
        if entry.get('source_observation'):
            saved=entry['source_observation'];queue=queue_for(case,fid,saved)
            # Uncertain thumbnail details must not force rebuilding the full
            # queue or discarding already recorded semantic observations.
            native={n for e in queue['evidence'] if e.get('native_size') for n in e['frames']}
            missing=sorted(set(stills)-native)
            if missing:
                if entry.get('route_handoff'):
                    raise ValueError('Source stage already closed; preserve original observation')
                from review_queue import build
                out=project.project_path(saved['queue']['path']).parent/('details-'+workflow.hash_json(missing)[:16])
                build(case,fid,str(out.relative_to(project.ROOT)),frames=missing)
                saved.setdefault('detail_queues',[]).append(artifact(out/'queue.json'))
            return progress(case,fid,entry)
        if entry.get('route_handoff') or entry.get('execution_units') or entry.get('material_handoff') or entry.get('current_candidate_id'):
            raise ValueError('Source stage already closed; preserve original handoff, do not rebuild observation materials')
        out=spec.get('out') or str((workflow.state_path(case).parent/'source-observation'/fid).relative_to(project.ROOT))
        result=build_pairs(case,fid,out,dict(source_only=True,stills=stills))
        path=project.project_path(result['inventory']);queue=project.read_json(path)
        queue['dependency']=workflow.requirement_binding(case,item)
        queue['operations']=item['operations']
        queue['user_input']=dict(path=case['user_input'],sha256=project.digest(project.project_path(case['user_input'])))
        if queue['paired_frames']!=list(range(entry['source_frame_count'])):
            raise ValueError('Source observation must decode every source frame')
        path.write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n')
        entry['source_observation']=dict(queue=artifact(path))
        result=progress(case,fid,entry)
    return result


def record(case, spec):
    from production_pipeline import require_production
    require_production(case)
    fid=spec['file_id'];item=workflow.item_for(case,fid)
    with workflow.locked_state(case) as state:
        workflow.verify_state_case(case,state);entry=state['files'][fid]
        semantic_execution = spec.get('from_semantic_scan')
        if semantic_execution:
            from semantic_scan import reviewed_packet
            spec = reviewed_packet(case,spec,entry)
        previous,_,_=verify(case,fid,entry)
        saved=entry['source_observation'];queue=queue_for(case,fid,saved)
        if (spec.get('queue_sha256')!=saved['queue']['sha256']
                or spec.get('dependency')!=workflow.requirement_binding(case,item)
                or not isinstance(spec.get('observer'),str) or not spec['observer'].strip()):
            raise ValueError('Source observation submission must bind queue, requirements and actual observer')
        submitted=spec.get('operations',{})
        if not submitted or set(submitted)-{o['id'] for o in item['operations']}:
            raise ValueError('Source observation submission has missing or unknown operations')
        operations={o['id']:dict(records=[],quality_requirements=[]) for o in item['operations']}
        if previous:operations=previous['operations']
        changed=False
        for oid,packet in submitted.items():
            rows=packet['records'];validate_rows(rows,queue)
            quality=packet.get('quality_requirements')
            if not isinstance(quality,list) or not quality or not all(isinstance(q,str) and q.strip() for q in quality):
                raise ValueError('Source observation requires operation quality requirements')
            old=operations[oid]
            if old['quality_requirements'] and quality!=old['quality_requirements']:
                raise ValueError('Source observation quality requirements already fixed')
            by_frame={r['frame']:r for r in old['records']}
            for row in rows:
                row=dict(row,observer=spec['observer']);n=row['frame'];parent=by_frame.get(n)
                if parent and parent!=row and parent['state']!='unresolved':
                    raise ValueError('Source observation already recorded; preserve original facts and diagnose corrections')
                changed=changed or parent!=row
                by_frame[n]=row
            operations[oid]=dict(records=[by_frame[n] for n in sorted(by_frame)],quality_requirements=quality)
        if not changed:
            return progress(case,fid,entry)
        if entry.get('route_handoff'):
            raise ValueError('Source observation is frozen; do not reopen source stage during production')
        summaries,_=summarize(item,operations)
        for oid,summary in summaries.items():operations[oid].update(summary)
        payload=dict(schema='source_observation_v1',file_id=fid,source_sha256=item['source_sha256'],
            dependency=workflow.requirement_binding(case,item),queue=saved['queue'],operations=operations,
            parent_receipt=saved.get('receipt'),accepted=False,
            scope='Actual host observation declaration, not machine semantic inference or business acceptance')
        if semantic_execution or (previous and previous.get('semantic_execution')):
            payload['semantic_execution'] = semantic_execution or previous['semantic_execution']
        path=project.project_path(saved['queue']['path']).parent/('receipt-'+workflow.hash_json(payload)[:20]+'.json')
        with path.open('x') as stream:json.dump(payload,stream,ensure_ascii=False,indent=2)
        saved['receipt']=artifact(path)
        result=progress(case,fid,entry)
    return result


def require_receipt(case, fid, entry, observation):
    if not required(case):return None
    receipt,_,_=verify(case,fid,entry,complete=True)
    if observation!=entry['source_observation']['receipt']:
        raise ValueError('Frozen source observation must consume the saved full-frame receipt')
    return receipt


def geometry_covers(source_instances, planned_instances):
    """One-to-one spatial consistency for explicit geometry, not recognition."""
    boxes=[]
    for inst in planned_instances:
        quad=inst.get('source_quad') or inst.get('word_quad')
        if not isinstance(quad,list) or len(quad)<3:
            return None
        if any(not isinstance(p,list) or len(p)!=2 or
               any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in quad):
            raise ValueError('Invalid source instance geometry in plan')
        boxes.append((min(p[0] for p in quad),min(p[1] for p in quad),
                      max(p[0] for p in quad),max(p[1] for p in quad)))
    matches={}
    def assign(index,seen):
        x,y,w,h=source_instances[index]['bbox']
        for j,(a,b,c,d) in enumerate(boxes):
            if j not in seen and max(x,a)<min(x+w,c) and max(y,b)<min(y+h,d):
                seen.add(j)
                if j not in matches or assign(matches[j],seen):
                    matches[j]=index
                    return True
        return False
    return all(assign(i,set()) for i in range(len(source_instances)))


def require_plan_instances(observation, operation_id, plan):
    """Catch known omissions, not unseen objects or incorrect semantic labels.

    Geometry plans expose separate instances. Whole-mask/provider plans with
    several targets need explicit semantic_instance_ids in the actual review;
    this declaration still needs visual verification, not an automatic pass.
    """
    if observation is None:return
    rows=plan['review']['records'];states=plan['review']['frame_states']
    source_rows=observation['operations'][operation_id]['records']
    if len(states)!=len(source_rows) or [r.get('frame') for r in rows]!=list(range(len(source_rows))):
        raise ValueError('Source instance plan needs every frame in source order')
    records={r['frame']:r for r in rows}
    for source in source_rows:
        n=source['frame'];expected=source['state'];state=states[n]
        if expected=='absent' and state!='absent':
            raise ValueError('Source observed absent but plan edits frame '+str(n))
        if expected=='occluded' and state not in ('occluded','absent'):
            raise ValueError('Fully occluded source target must not be edited at '+str(n))
        if expected=='visible':
            if state not in ('visible','partial'):
                raise ValueError('Observed visible instance missing from plan frame '+str(n))
            row=records.get(n,{})
            instances=row.get('instances')
            planned=[i for i in instances if i.get('state',state) in ('visible','partial')] if isinstance(instances,list) else [row] if row else []
            count=len(planned)
            geometry=geometry_covers(source['instances'],planned)
            if geometry is False:
                raise ValueError('Observed source instance position missing from plan at '+str(n))
            if len(source['instances'])>1 and geometry is None:
                ids=row.get('semantic_instance_ids',[])
                if set(ids)!={i['id'] for i in source['instances']} or len(ids)!=len(set(ids)):
                    raise ValueError('Observed source instances need explicit per-frame plan coverage at '+str(n))
            elif count!=len(source['instances']):
                raise ValueError('Observed source instance count differs from plan at '+str(n))
