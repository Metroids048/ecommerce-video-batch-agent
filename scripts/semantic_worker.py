"""Direct offline MLX-VLM worker; no HTTP service, downloads or model fallback."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time


def memory_event(stage, frame=None, result=None):
    import mlx.core as mx
    data=dict(event='mlx_memory',stage=stage,frame=frame,at=time.time(),
              active_bytes=mx.get_active_memory(),cache_bytes=mx.get_cache_memory(),peak_bytes=mx.get_peak_memory())
    if result is not None:
        data.update({k:getattr(result,k,None) for k in ('prompt_tokens','generation_tokens','prompt_tps','generation_tps')})
    print(json.dumps(data),file=sys.stderr,flush=True)


def deny_network(*args, **kwargs):
    raise RuntimeError('Semantic worker is local only; network is disabled')


def imports():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    return load, generate, apply_chat_template


def probe():
    imports()
    import mlx.core
    import mlx_vlm.utils
    import mlx_vlm.prompt_utils
    import mlx_vlm.structured
    import llguidance.hf
    modules = (mlx.core, mlx_vlm.utils, mlx_vlm.prompt_utils, mlx_vlm.structured)
    return dict(versions={n:importlib.metadata.version(n) for n in ('mlx','mlx-vlm','transformers','Pillow')},
                modules={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules})


def response_schema(operation_ids):
    prediction = dict(type='object',additionalProperties=False,
        required=['state','all_instances','reason','instances'],properties=dict(
            state=dict(type='string',enum=['visible','absent','occluded','unresolved']),
            all_instances=dict(type='boolean'),reason=dict(type='string',minLength=1),
            instances=dict(type='array',items=dict(type='object',additionalProperties=False,required=['box'],
                properties=dict(box=dict(type='array',minItems=4,maxItems=4,
                    items=dict(type='number',minimum=0,maximum=1000)))))))
    return dict(type='object',additionalProperties=False,required=['operations'],properties=dict(
        operations=dict(type='object',additionalProperties=False,required=operation_ids,
                        properties={oid:prediction for oid in operation_ids})))


def main():
    if sys.argv[1:] == ['--probe']:
        with contextlib.redirect_stdout(sys.stderr):
            result = probe()
        print(json.dumps(result), flush=True)
        return
    # stdout is reserved for one reply per frame; library diagnostics go to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        load, generate, template = imports()
        init = json.loads(sys.stdin.readline())
        model_path = Path(init['model_path'])
        if not model_path.is_dir():
            raise ValueError('An existing local model directory is required')
        model, processor = load(str(model_path), local_files_only=True, trust_remote_code=False, token=False)
        memory_event('loaded')
        from mlx_vlm.structured import build_json_schema_logits_processor
        tokenizer = processor.tokenizer if hasattr(processor,'tokenizer') else processor
        schema = response_schema(init['operation_ids'])
    print(json.dumps(dict(ready=True)), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            memory_event('before_generate',request['frame'])
            prompt = template(processor, model.config, request['prompt'], num_images=1, enable_thinking=False)
            try:
                result = generate(model=model, processor=processor, prompt=prompt, image=[request['image']],
                                  max_tokens=2048, temperature=0, verbose=False,
                                  logits_processors=[build_json_schema_logits_processor(tokenizer,schema)])
            except Exception:
                memory_event('failed',request['frame'])
                raise
            memory_event('after_generate',request['frame'],result)
        print(json.dumps(dict(frame=request['frame'], text=result.text)), flush=True)


if __name__ == '__main__':
    main()
