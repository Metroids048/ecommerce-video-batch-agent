"""Exact local startup checkpoint. File hashes are not proof of a host's reasoning."""
import os
import project
import workflow


def capture(case):
    paths=sorted(project.ROOT.glob('scripts/*.py'))+sorted(project.ROOT.glob('scripts/*.mjs'))
    paths+=sorted(project.ROOT.glob('prompts/*.md'))+sorted(project.ROOT.glob('.agents/skills/*/SKILL.md'))
    paths+=sorted(project.ROOT.glob('web/*.html'))
    model_path=project.ROOT/'config/models.json'
    if model_path.is_file():paths.append(model_path)
    hashes={str(p.relative_to(project.ROOT)):project.digest(p) for p in paths}
    return dict(captured_at=workflow.now(),pid=os.getpid(),case_id=case['batch_id'],
        runtime_fingerprint=workflow.hash_json(hashes),files=hashes,
        scope='Files at process/checkpoint capture; Codex observation and prompt reading are separately attested',
        model_configuration_present=model_path.is_file(),
        remote_policy=project.read_json(model_path).get('remote_calls_allowed',False) if model_path.is_file() else False)
