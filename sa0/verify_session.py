"""Read and verify an integration rehearsal; never launches additional compute."""
import argparse
import json
from pathlib import Path

from sa0 import control


def verify(session):
    state, protocol = control.load_session(session)
    names = ['baseline','agent_candidate']
    trials = {name:control.read(session/'trials'/name/'result.json') for name in names}
    traces = {name:[json.loads(line) for line in (session/'trials'/name/'training.jsonl').read_text().splitlines()] for name in names}
    for field in ['sample_ids','lr','step']:
        if [row[field] for row in traces['baseline']] != [row[field] for row in traces['agent_candidate']]:
            raise ValueError(f'Mismatched training control: {field}')
    for relative in control.MUTABLE:
        if control.sha(control.ROOT/relative)!=state['initial_workspace_sha256'][relative]:
            raise ValueError('Original training source changed')
    for name,record in trials.items():
        if control.sha(session/'trials'/name/'checkpoint.pt')!=record['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
    if control.git(session/'workspace','status','--porcelain'):
        raise ValueError('Isolated workspace is dirty')
    control.check_paths(state['initial_workspace_sha256'],control.files(session/'workspace'))
    agent_protocol = control.read(session/'agent/protocol.json')
    if control.sha(control.ROOT/'sa0/agent_loop.py')!=agent_protocol['source_sha256']:
        raise ValueError('Model integration source changed')
    ledgers = {name:control.read(session/'agent'/name/'ledger.json') for name in ['01_proposal','02_feedback']}
    if any(record['error'] is not None for record in ledgers.values()):
        raise ValueError('Model call was not successful')
    decision = control.read(session/'decision.json')
    summary = control.read(session/'agent/summary.json')
    if summary['decision']!=decision:
        raise ValueError('Agent summary decision mismatch')
    gpu_wall = sum(control.read(session/'trials'/name/'ledger.json')['charged_wall_seconds'] for name in names)
    if gpu_wall>protocol['session_compute_wall_seconds']:
        raise ValueError('Compute budget exceeded')
    result = {'matched_training_sample_and_lr_sequences':True,
              'original_training_sources_unchanged':True,'checkpoint_hashes_match':True,
              'frozen_data_and_trusted_sources_match':True,'isolated_workspace_clean':True,
              'integration_source_hash_match':True,'completed_model_calls':len(ledgers),
              'model_usage':{name:record['usage'] for name,record in ledgers.items()},
              'model_call_wall_seconds':sum(record['elapsed_seconds'] for record in ledgers.values()),
              'gpu_process_wall_seconds':gpu_wall,'monetary_cost':None,
              'formal_independent_sessions_completed':0}
    control.write_new(session/'agent/verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--session',type=Path,required=True)
    verify(parser.parse_args().session.resolve())
