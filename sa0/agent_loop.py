"""Windows bridge: one model proposal, one controlled GPU trial, model feedback."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from sa0 import control

ROOT = control.ROOT
HYPOTHESIS_KEYS = ['hypothesis','change','expected_effect','risk','controlled_variables']


def object_schema(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


PROPOSAL_SCHEMA = object_schema({
    **{key:{'type':'string'} for key in HYPOTHESIS_KEYS if key!='controlled_variables'},
    'controlled_variables':{'type':'array','items':{'type':'string'}},
    'plan':{'type':'array','items':{'type':'string'}},
    'edits':{'type':'array','items':object_schema({
        'path':{'type':'string','enum':['train/experiment.py']},
        'old':{'type':'string'},'new':{'type':'string'}})},
})
FEEDBACK_SCHEMA = object_schema({
    'interpretation':{'type':'string'},
    'hypothesis_supported':{'type':'string','enum':['inconclusive','contradicted','limited_support']},
    'limitations':{'type':'array','items':{'type':'string'}},
    'next_hypothesis':{'type':'string'},
    'next_validation':{'type':'string'},
})


def wsl_path(path):
    path = Path(path).resolve()
    if len(path.drive)!=2 or path.drive[1]!=':':
        raise ValueError('Use a local Windows drive path for this WSL bridge')
    return '/mnt/'+path.drive[0].lower()+'/'+path.as_posix()[3:]


def validate_edits(workspace, proposal):
    if set(proposal) != set(PROPOSAL_SCHEMA['properties']):
        raise ValueError('Unexpected proposal fields')
    if any(not proposal[key] for key in HYPOTHESIS_KEYS+['plan','edits']):
        raise ValueError('Incomplete hypothesis or plan')
    edits = proposal['edits']
    if not 1<=len(edits)<=3:
        raise ValueError('Allow one to three exact replacements for this demo')
    original = (workspace/'train/experiment.py').read_text(encoding='utf-8')
    updated = original
    for edit in edits:
        if set(edit)!= {'path','old','new'} or edit['path']!='train/experiment.py':
            raise ValueError('Only train/experiment.py edits are enabled in this demo')
        if not edit['old'] or edit['old']==edit['new'] or updated.count(edit['old'])!=1:
            raise ValueError('Each old snippet must match exactly once and produce a change')
        updated = updated.replace(edit['old'],edit['new'],1)
    if len(updated)>len(original)+6000:
        raise ValueError('Candidate is too large for this small demonstration')
    tree = ast.parse(updated)
    # First demo supports changing a loss or optimizer knob, not adding executable modules.
    before_tree = ast.parse(original)
    if [ast.dump(n) for n in tree.body if not isinstance(n,ast.FunctionDef)] != [ast.dump(n) for n in before_tree.body if not isinstance(n,ast.FunctionDef)]:
        raise ValueError('Imports and module-level code must remain unchanged')
    before_functions = {n.name:n for n in before_tree.body if isinstance(n,ast.FunctionDef)}
    after_functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    if set(before_functions)!=set(after_functions):
        raise ValueError('Do not add or remove training interfaces')
    changes = [name for name in before_functions if ast.dump(before_functions[name])!=ast.dump(after_functions[name])]
    if len(changes)!=1 or changes[0] not in ['compute_loss','build_optimizer']:
        raise ValueError('Change exactly one loss or optimizer function')
    for name in before_functions:
        if ast.dump(before_functions[name].args)!=ast.dump(after_functions[name].args):
            raise ValueError('Keep function signatures unchanged')
    return original,updated


def apply_edits(workspace, proposal):
    original, updated = validate_edits(workspace,proposal)
    # Validate every edit before writing any file.
    path = workspace/'train/experiment.py'
    if path.read_text(encoding='utf-8')!=original:
        raise ValueError('Source changed during proposal validation')
    path.write_text(updated,encoding='utf-8',newline='\n')


def model_call(folder, name, prompt, schema, codex):
    output = folder/name; output.mkdir(exist_ok=False)
    (output/'prompt.txt').write_text(prompt,encoding='utf-8')
    control.write_new(output/'schema.json',schema)
    command = [codex,'exec','--ignore-user-config','--ephemeral','--json','--color','never',
               '--sandbox','read-only','--skip-git-repo-check','-C',str(folder/'context'),
               '-c','web_search="disabled"','-c','features.shell_tool=false',
               '-c','features.unified_exec=false','-c','features.multi_agent=false',
               '-c','features.hooks=false','-c','features.memories=false',
               '-c','features.plugins=false','-c','approval_policy="never"',
               '--output-schema',str(output/'schema.json'),
               '--output-last-message',str(output/'response.json'),'-']
    control.write_new(output/'request.json',{'command':command,'timeout_seconds':180,
                                            'authentication':'existing ChatGPT CLI login',
                                            'model':'CLI default; no model override',
                                            'external_retrieval':'disabled'})
    started = time.perf_counter(); error = None; usage = None
    try:
        with (output/'events.jsonl').open('x',encoding='utf-8') as events, (output/'stderr.log').open('x',encoding='utf-8') as stderr:
            subprocess.run(command,input=prompt,encoding='utf-8',stdout=events,stderr=stderr,
                           timeout=180,check=True)
        records = [json.loads(line) for line in (output/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        usages = [r.get('usage') for r in records if r.get('type')=='turn.completed']
        usage = usages[-1] if usages else None
        forbidden = [r for r in records if r.get('type','').startswith('item.') and
                     r.get('item',{}).get('type') in ['command_execution','web_search','mcp_tool_call','file_change']]
        if forbidden:
            raise ValueError('Tool activity detected in a context-only model call')
        response = control.read(output/'response.json')
        if set(response)!=set(schema['properties']):
            raise ValueError('Model output does not match the requested top-level fields')
        return response
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
        raise
    finally:
        control.write_new(output/'ledger.json',{'elapsed_seconds':time.perf_counter()-started,
                     'usage':usage,'monetary_cost':None,'error':error,
                     'note':'CLI reported tokens where available; no monetary billing measurement.'})


def run(session, data_root, wsl_python):
    if sys.platform!='win32':
        raise RuntimeError('This launcher currently runs on Windows and trains in WSL')
    if session.exists():
        raise FileExistsError('Use a new session; existing experiments are never overwritten')
    codex = shutil.which('codex')
    if not codex:
        raise FileNotFoundError('Codex CLI is not on PATH')
    status = subprocess.run([codex,'login','status'],capture_output=True,text=True,timeout=15)
    if status.returncode:
        raise RuntimeError('Codex CLI needs an existing login; no credentials are read by this launcher')
    def controller(*args):
        command = ['wsl','-e',wsl_python,'-m','sa0.control',*args]
        subprocess.run(command,cwd=ROOT,check=True,timeout=150)
    controller('init','--session',wsl_path(session),'--data-root',wsl_path(data_root),'--steps','5')
    agent = session/'agent'; agent.mkdir()
    (agent/'context').mkdir()
    control.write_new(agent/'protocol.json',{'max_model_calls':2,'max_seconds_per_call':180,
        'code_edit_authority':'controller applies validated exact replacements',
        'first_demo_edit_surface':'one compute_loss or build_optimizer change',
        'source_sha256':control.sha(Path(__file__)),
        'scope':'automatic one-candidate integration rehearsal; not formal SA0',
        'formal_independent_sessions_completed':0})
    overall_started = time.perf_counter(); error = None
    try:
        print('Running matched baseline...',flush=True)
        controller('run','--session',wsl_path(session),'--name','baseline','--role','baseline')
        baseline = control.read(session/'trials/baseline/result.json')
        code = (session/'workspace/train/experiment.py').read_text(encoding='utf-8')
        protocol = control.read(session/'protocol.json')
        repo_card = ('MAVI: feature tensor B,1,24,256,256; output B,256,256. '
                    'Feature channels: 4 static power maps then 20 temporal maps. '
                    'Target is the preprocessed log-scaled IR-drop image. '
                    'MAE is float; official NRMS/SSIM use clipped 8-bit images. '
                    'Protected train_batch checks finite loss and gradients. '
                    'AdamW lr=2e-4, weight_decay=1e-2; cosine horizon 200000. '
                    '3 training images and 1 validation image; not a generalization benchmark.')
        prompt = ('You are the single research agent for an IR predictor integration rehearsal. '
                  'Use only the supplied context. Do not call tools, browse, read files, or modify files. '
                  'Propose ONE modest, scientifically motivated loss OR optimizer change. '
                  'Do not change model, schedule, train_batch, seeds, data, metrics, or budget. '
                  'Do not optimize for a particular sample name or claim five steps proves superiority. '
                  'Return the schema JSON, including hypothesis, risk, controls, implementation plan, '
                  'and 1-3 exact old/new replacements in train/experiment.py. '
                  'Change exactly one function: compute_loss OR build_optimizer. '
                  'Preserve all interfaces, shape checks and finite-value checks. '
                  'No new imports, module-level code, tools or subprocesses. '
                  'Explanations should be in Chinese. No paper citations from memory are required.\n'
                  f'REPOSITORY CARD:\n{repo_card}\nPROTOCOL:\n{json.dumps(protocol)}\n'
                  f'BASELINE VALIDATION:\n{json.dumps(baseline["evaluation"])}\n'
                  f'BASELINE TRAINING:\n{(session/"trials/baseline/training.jsonl").read_text()}\n'
                  f'TRAINING SOURCE:\n{code}')
        print('Calling model for hypothesis and exact edits...',flush=True)
        proposal = model_call(agent,'01_proposal',prompt,PROPOSAL_SCHEMA,codex)
        validate_edits(session/'workspace',proposal)
        control.write_new(agent/'hypothesis.json',{key:proposal[key] for key in HYPOTHESIS_KEYS})
        controller('propose','--session',wsl_path(session),'--hypothesis',wsl_path(agent/'hypothesis.json'))
        apply_edits(session/'workspace',proposal)
        controller('commit','--session',wsl_path(session))
        print('Running model-generated candidate...',flush=True)
        controller('run','--session',wsl_path(session),'--name','agent_candidate','--role','candidate')
        controller('decide','--session',wsl_path(session))
        decision = control.read(session/'decision.json')
        candidate = control.read(session/'trials/agent_candidate/result.json')
        feedback = ('You are the same single IR research agent continuing the recorded proposal. '
                    'Use only this supplied history. Do not call tools, browse, modify files or launch more trials. '
                    'Analyze measured evidence, do not invent improvements or ignore guard failures. '
                    'The controller decision is authoritative. Five steps and one validation image are inconclusive '
                    'about generalization. Different loss values need not be comparable. '
                    'Return Chinese interpretation, hypothesis_supported, limitations, next_hypothesis '
                    'and next_validation. The next idea is recorded only; budget forbids executing it.\n'
                    f'PRIOR PROPOSAL:\n{json.dumps(proposal,ensure_ascii=False)}\n'
                    f'BASELINE:\n{json.dumps(baseline)}\nCANDIDATE:\n{json.dumps(candidate)}\n'
                    f'DECISION:\n{json.dumps(decision)}\n')
        print('Returning measured results to the model...',flush=True)
        reflection = model_call(agent,'02_feedback',feedback,FEEDBACK_SCHEMA,codex)
        control.write_new(agent/'summary.json',{'proposal':proposal,'decision':decision,'reflection':reflection,
            'completed_model_calls':2,'applied_candidates':1,'formal_independent_sessions_completed':0})
        report = ('# 模型接入闭环演练\n\n'
                  '已完成模型提出假设和修改 → 控制器执行 → 统一验证 → 模型分析结果。\n'
                  '本轮借鉴 AuDoPEDA 的阶段划分，并未复现其 Docmaker、代码图、DSPy 或文献检索。\n'
                  '两个调用由同一后端完成，以完整历史传递上下文，不是两个独立研究 agent。\n\n'
                  f'## 假设\n\n{proposal["hypothesis"]}\n\n修改：{proposal["change"]}\n\n'
                  f'## 实测决定\n\n{decision["decision"]}\n\n'
                  f'基线指标：{json.dumps(decision["baseline_metrics"])}\n\n'
                  f'候选指标：{json.dumps(decision["candidate_metrics"])}\n\n'
                  f'## 模型反馈\n\n{reflection["interpretation"]}\n\n'
                  + '\n'.join('- '+item for item in reflection['limitations'])
                  + f'\n\n下一假设（未执行）：{reflection["next_hypothesis"]}\n\n'
                  '本轮只验证接入流程，正式 SA0 次数为 0。模型调用使用已有 CLI 登录；费用未知。\n')
        (agent/'REPORT.md').write_text(report,encoding='utf-8')
        print(f'Integration complete. Report: {agent/"REPORT.md"}',flush=True)
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
        raise
    finally:
        control.write_new(agent/'loop_ledger.json',{'status':'completed' if error is None else 'failed',
            'elapsed_seconds':time.perf_counter()-overall_started,'error':error})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--wsl-python',default='/home/laopier/miniconda3/envs/circuitnet/bin/python')
    args = parser.parse_args()
    run(args.session.resolve(),args.data_root.resolve(),args.wsl_python)


if __name__=='__main__':
    main()
