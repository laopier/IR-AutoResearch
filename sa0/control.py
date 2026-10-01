from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MUTABLE = {'train/mavi.py', 'train/experiment.py'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def files(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        relative = path.relative_to(root)
        if '.git' in relative.parts or '__pycache__' in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink is not permitted in workspace: {relative}')
        if path.is_file():
            result[relative.as_posix()] = sha(path)
    return result


def check_paths(expected, actual):
    missing = set(expected) - set(actual)
    added = set(actual) - set(expected)
    forbidden = {p for p in expected if p not in MUTABLE and expected[p] != actual.get(p)}
    if missing or added or forbidden:
        raise ValueError(f'Workspace boundary violated: missing={sorted(missing)}, added={sorted(added)}, protected_changes={sorted(forbidden)}')
    return sorted(p for p in MUTABLE if actual.get(p) != expected.get(p))


def git(workspace, *args):
    return subprocess.check_output(['git', '-c', f'safe.directory={workspace}', *args],
                                   cwd=workspace, text=True).strip()


def commit(workspace, message):
    git(workspace, 'add', '.')
    git(workspace, '-c', 'user.name=SA0 Harness', '-c', 'user.email=sa0-harness@localhost',
        'commit', '-m', message)
    return git(workspace, 'rev-parse', 'HEAD')


def event(session, kind, payload):
    record = {'event': kind, 'unix_time': time.time(), **payload}
    with (session/'events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')


def load_session(session):
    state = read(session/'session.json')
    if sha(session/'protocol.json') != state['protocol_sha256']:
        raise ValueError('Protocol changed after session initialization')
    protocol = read(session/'protocol.json')
    for name, digest in state['trusted_source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Trusted controller/evaluator source changed: {name}')
    for name, digest in state['data_sha256'].items():
        if sha(session/name) != digest:
            raise ValueError(f'Frozen rehearsal data changed: {name}')
    return state, protocol


def init(session, data_root, steps):
    policy = read(ROOT/'program/policy.json')['stages']['smoke']
    if not 1 <= steps <= policy['max_steps']:
        raise ValueError('Requested steps exceed existing smoke policy')
    manifests = {'train':ROOT/'prepare/manifests/smoke_train.csv',
                 'validation':ROOT/'prepare/manifests/smoke_validation.csv'}
    for name, path in manifests.items():
        if sha(path) != policy[f'{name}_manifest_sha256']:
            raise ValueError(f'Smoke manifest hash mismatch: {name}')
    # Validate manifest syntax before creating the session.
    from prepare.split_manifest import read_manifest
    rows = {name:read_manifest(path) for name,path in manifests.items()}
    all_paths = {}
    for name, pairs in rows.items():
        all_paths[name] = []
        for pair in pairs:
            for relative in pair:
                source = (data_root/relative).resolve()
                source.relative_to(data_root.resolve())
                if not source.is_file():
                    raise FileNotFoundError(source)
                all_paths[name].append((relative, source))
    if {p for _,p in all_paths['train']} & {p for _,p in all_paths['validation']}:
        raise ValueError('Train/validation data overlap')
    session.mkdir(parents=True, exist_ok=False)
    workspace = session/'workspace'; workspace.mkdir()
    for folder in ['prepare', 'program', 'train', 'tests']:
        shutil.copytree(ROOT/folder, workspace/folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT/'CIRCUITNET_LICENSE', workspace/'CIRCUITNET_LICENSE')
    (workspace/'.gitignore').write_text('__pycache__/\n*.py[cod]\n', encoding='utf-8')
    frozen_hashes = files(workspace)
    git(workspace, 'init')
    baseline_commit = commit(workspace, 'Freeze mini SA0 starting snapshot')
    data_hashes = {}
    for name, pairs in all_paths.items():
        destination = session/f'data_{name}'
        destination.mkdir()
        manifest = destination/'manifest.csv'; shutil.copy2(manifests[name], manifest)
        data_hashes[manifest.relative_to(session).as_posix()] = sha(manifest)
        for relative, source in pairs:
            target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            data_hashes[target.relative_to(session).as_posix()] = sha(target)
    protocol = {'stage':'mini_rehearsal', 'performance_claims_allowed':False,
                'mutable_paths':sorted(MUTABLE), 'steps':steps, 'batch_size':1, 'seed':0,
                'lr_horizon_steps':200000, 'max_trials':2, 'trial_timeout_seconds':120,
                'session_compute_wall_seconds':240, 'max_candidates':1,
                'selection_rule':'strictly lower mae; NRMS not higher, SSIM not lower; parameters and peak memory <= 1.10x baseline',
                'selection_scope':'demo only; not a formal search threshold',
                'network_policy':'offline; Python audit guards in workers; not OS sandbox',
                'validation_policy':'separate evaluation process; validation labels absent from training view',
                'agent_api_tokens':None, 'agent_api_cost':None,
                'api_accounting_note':'Desktop agent usage not measured; do not report zero cost.'}
    write_new(session/'protocol.json', protocol)
    trusted_names = ['sa0/control.py','sa0/worker.py','prepare/evaluator.py','program/policy.json']
    state = {'baseline_commit':baseline_commit, 'initial_workspace_sha256':frozen_hashes,
             'protocol_sha256':sha(session/'protocol.json'), 'data_sha256':data_hashes,
             'trusted_source_sha256':{p:sha(ROOT/p) for p in trusted_names},
             'origin_git_commit':git(ROOT,'rev-parse','HEAD'),
             'origin_git_dirty':bool(git(ROOT,'status','--porcelain'))}
    write_new(session/'session.json', state)
    event(session,'session_started',{'baseline_commit':baseline_commit,'steps':steps})
    print(f'Session created: {session}\nEditable workspace: {workspace}', flush=True)


def propose(session, hypothesis):
    state, _ = load_session(session)
    workspace = session/'workspace'
    if files(workspace) != state['initial_workspace_sha256'] or git(workspace,'status','--porcelain'):
        raise ValueError('Record the hypothesis before changing baseline code')
    baseline = session/'trials/baseline/ledger.json'
    if not baseline.exists() or read(baseline)['status'] != 'completed':
        raise ValueError('Complete the baseline before proposing a candidate')
    proposal = read(hypothesis)
    for key in ['hypothesis','change','expected_effect','risk','controlled_variables']:
        if not proposal.get(key):
            raise ValueError(f'Hypothesis must document {key}')
    write_new(session/'proposal.json', proposal)
    write_new(session/'proposal_receipt.json', {'sha256':sha(session/'proposal.json'),
                                              'before_commit':git(workspace,'rev-parse','HEAD')})
    event(session,'hypothesis_registered',{'sha256':sha(session/'proposal.json')})


def commit_candidate(session):
    state, _ = load_session(session)
    proposal = read(session/'proposal_receipt.json')
    if sha(session/'proposal.json') != proposal['sha256']:
        raise ValueError('Registered hypothesis changed')
    workspace = session/'workspace'
    changed = check_paths(state['initial_workspace_sha256'], files(workspace))
    if not changed or git(workspace,'rev-parse','HEAD') != proposal['before_commit']:
        raise ValueError('Candidate must change allowed files from the registered starting commit')
    head = commit(workspace,'SA0 candidate: '+read(session/'proposal.json')['change'])
    event(session,'candidate_committed',{'commit':head,'changed_paths':changed})
    print(head)


def run_trial(session, name, role):
    if not name.replace('_','').replace('-','').isalnum():
        raise ValueError('Trial name must contain only letters, numbers, underscores or hyphens')
    state, protocol = load_session(session)
    workspace = session/'workspace'
    changed = check_paths(state['initial_workspace_sha256'], files(workspace))
    if git(workspace,'status','--porcelain'):
        raise ValueError('Commit candidate changes in the isolated workspace before running')
    trials_dir = session/'trials'; trials_dir.mkdir(exist_ok=True)
    trials = sorted(p for p in trials_dir.iterdir() if p.is_dir())
    if len(trials) >= protocol['max_trials']:
        raise ValueError('Trial count budget exhausted, including failures')
    used = sum(read(p/'ledger.json')['charged_wall_seconds'] for p in trials if (p/'ledger.json').exists())
    if any(not (p/'ledger.json').exists() for p in trials):
        raise ValueError('An interrupted trial needs reconciliation before more compute')
    remaining = protocol['session_compute_wall_seconds'] - used
    if remaining <= 0:
        raise ValueError('Session compute budget exhausted')
    if role == 'baseline':
        if changed or trials:
            raise ValueError('Baseline must be the first trial using frozen code')
        hypothesis_record = {'hypothesis':'Fixed baseline; no code changes.'}
    else:
        if not changed or len(trials) != 1 or read(trials[0]/'ledger.json')['status'] != 'completed':
            raise ValueError('Candidate requires a completed baseline and mutable-path changes')
        receipt = read(session/'proposal_receipt.json')
        if sha(session/'proposal.json') != receipt['sha256']:
            raise ValueError('Registered hypothesis changed')
        hypothesis_record = read(session/'proposal.json')
    output = trials_dir/name; output.mkdir(exist_ok=False)
    head = git(workspace,'rev-parse','HEAD')
    before = files(workspace)
    write_new(output/'hypothesis.json',hypothesis_record)
    (output/'code.diff').write_text(git(workspace,'diff',state['baseline_commit'],head),encoding='utf-8')
    write_new(output/'source.json',{'commit':head,'dirty':False,'file_sha256':before,
                                  'changed_paths':changed,'role':role})
    event(session,'trial_started',{'trial':name,'role':role,'commit':head})
    started = time.perf_counter()
    status = 'failed'; error = None
    try:
        timeout = min(protocol['trial_timeout_seconds'],remaining)
        for mode in ['train','evaluate']:
            if mode == 'evaluate':
                if files(workspace) != before:
                    raise ValueError('Workspace changed during training')
            data = session/('data_train' if mode == 'train' else 'data_validation')
            command = [sys.executable,str(ROOT/'sa0/worker.py'),'--workspace',str(workspace),
                       '--mode',mode,'--data-root',str(data),'--manifest',str(data/'manifest.csv'),
                       '--output',str(output),'--steps',str(protocol['steps']),
                       '--batch-size',str(protocol['batch_size']),'--seed',str(protocol['seed'])]
            if mode == 'evaluate':
                command += ['--checkpoint',str(output/'checkpoint.pt')]
            available = timeout-(time.perf_counter()-started)
            if available <= 0:
                raise TimeoutError('Trial wall-time budget exhausted')
            with (output/f'{mode}.log').open('x',encoding='utf-8') as log:
                subprocess.run(command,cwd=workspace,stdout=log,stderr=subprocess.STDOUT,
                               timeout=available,check=True,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        if files(workspace) != before or git(workspace,'rev-parse','HEAD') != head or git(workspace,'status','--porcelain'):
            raise ValueError('Workspace changed during execution')
        load_session(session)
        training = read(output/'train.json'); evaluation = read(output/'evaluate.json')
        if training['steps_completed'] != protocol['steps']:
            raise ValueError('Worker completed a different training budget')
        if set(evaluation['metrics']) != {'mae_float','nrms_official','ssim_official'} or not all(math.isfinite(v) for v in evaluation['metrics'].values()):
            raise ValueError('Invalid evaluation metrics')
        write_new(output/'result.json',{'role':role,'commit':head,'protocol':protocol,
                  'train':training,'evaluation':evaluation,'checkpoint_sha256':sha(output/'checkpoint.pt')})
        status = 'completed'
        print(f'{name} completed: {evaluation["metrics"]}',flush=True)
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
        raise
    finally:
        seconds = time.perf_counter()-started
        write_new(output/'ledger.json',{'status':status,'charged_wall_seconds':seconds,
                                       'charged_single_gpu_wall_hours':seconds/3600,'error':error})
        event(session,'trial_finished',{'trial':name,'status':status,'charged_wall_seconds':seconds,'error':error})


def decide(session):
    state, protocol = load_session(session)
    trials = [p for p in (session/'trials').iterdir() if p.is_dir()]
    completed = [read(p/'result.json') for p in trials if (p/'result.json').exists()]
    baseline = next(r for r in completed if r['role']=='baseline')
    candidate = next(r for r in completed if r['role']=='candidate')
    b = baseline['evaluation']['metrics']; c = candidate['evaluation']['metrics']
    checks = {'mae_improved':c['mae_float']<b['mae_float'],
              'nrms_not_worse':c['nrms_official']<=b['nrms_official'],
              'ssim_not_worse':c['ssim_official']>=b['ssim_official'],
              'parameter_guard':candidate['train']['parameter_count']<=1.1*baseline['train']['parameter_count'],
              'memory_guard':candidate['train']['peak_allocated_mib']<=1.1*baseline['train']['peak_allocated_mib']}
    keep = all(checks.values())
    workspace = session/'workspace'
    check_paths(state['initial_workspace_sha256'],files(workspace))
    if git(workspace,'rev-parse','HEAD') != candidate['commit'] or git(workspace,'status','--porcelain'):
        raise ValueError('Candidate workspace is no longer at evaluated clean commit')
    candidate_folder = next(p for p in trials if (p/'result.json').exists() and read(p/'result.json')['role']=='candidate')
    if files(workspace) != read(candidate_folder/'source.json')['file_sha256']:
        raise ValueError('Candidate files differ from the evaluated snapshot')
    decision = {'decision':'keep_for_rehearsal' if keep else 'rollback_to_baseline',
                'checks':checks,'baseline_metrics':b,'candidate_metrics':c,
                'selected_commit':candidate['commit'] if keep else baseline['commit'],
                'scope':'mini flow demonstration; not evidence of model superiority',
                'independent_autoresearch_sessions_completed':0}
    write_new(session/'decision.json',decision)
    if not keep:
        for relative in sorted(MUTABLE):
            content = subprocess.check_output(['git','-c',f'safe.directory={workspace}',
                                               'show',f'{state["baseline_commit"]}:{relative}'],cwd=workspace)
            (workspace/relative).write_bytes(content)
        commit(workspace,'Restore frozen baseline after mini candidate rejection')
    event(session,'decision',decision)
    print(json.dumps(decision,indent=2,ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action',required=True)
    create = commands.add_parser('init'); create.add_argument('--session',type=Path,required=True)
    create.add_argument('--data-root',type=Path,required=True); create.add_argument('--steps',type=int,default=5)
    trial = commands.add_parser('run'); trial.add_argument('--session',type=Path,required=True)
    trial.add_argument('--name',required=True); trial.add_argument('--role',choices=['baseline','candidate'],required=True)
    proposal = commands.add_parser('propose'); proposal.add_argument('--session',type=Path,required=True)
    proposal.add_argument('--hypothesis',type=Path,required=True)
    candidate_commit = commands.add_parser('commit'); candidate_commit.add_argument('--session',type=Path,required=True)
    select = commands.add_parser('decide'); select.add_argument('--session',type=Path,required=True)
    args = parser.parse_args(); args.session = args.session.resolve()
    if args.action=='init': init(args.session,args.data_root.resolve(),args.steps)
    elif args.action=='run': run_trial(args.session,args.name,args.role)
    elif args.action=='propose': propose(args.session,args.hypothesis)
    elif args.action=='commit': commit_candidate(args.session)
    else: decide(args.session)


if __name__=='__main__': main()
