import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

from sa0 import control


class BoundaryTests(unittest.TestCase):
    def test_protected_extra_and_deleted_files_are_rejected(self):
        expected = {'train/experiment.py':'old', 'prepare/evaluator.py':'fixed'}
        for actual in [
            {'train/experiment.py':'new', 'prepare/evaluator.py':'tampered'},
            {**expected, 'train/shortcut.py':'new'},
            {'train/experiment.py':'old'},
        ]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                control.check_paths(expected, actual)
        self.assertEqual(control.check_paths(expected, {**expected, 'train/experiment.py':'new'}), ['train/experiment.py'])

    def test_existing_records_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            control.write_new(path, {'original':True})
            with self.assertRaises(FileExistsError):
                control.write_new(path, {'original':False})
            self.assertEqual(control.read(path), {'original':True})

    def test_failed_trials_exhaust_count_budget_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory); workspace = session/'workspace'; workspace.mkdir()
            for name in ['baseline','failed_candidate']:
                folder = session/'trials'/name; folder.mkdir(parents=True)
                control.write_new(folder/'ledger.json', {'status':'failed','charged_wall_seconds':1})
            with patch.object(control,'load_session',return_value=({'initial_workspace_sha256':{}},{'max_trials':2})), patch.object(control,'git',return_value=''):
                with self.assertRaisesRegex(ValueError,'count budget'):
                    control.run_trial(session,'third','candidate')

    def test_interrupted_trial_blocks_unaccounted_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory); (session/'workspace').mkdir()
            (session/'trials/interrupted').mkdir(parents=True)
            with patch.object(control,'load_session',return_value=({'initial_workspace_sha256':{}},{'max_trials':2})), patch.object(control,'git',return_value=''):
                with self.assertRaisesRegex(ValueError,'reconciliation'):
                    control.run_trial(session,'retry','candidate')

    def test_proposal_must_precede_code_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory); (session/'workspace').mkdir()
            with patch.object(control,'load_session',return_value=({'initial_workspace_sha256':{'train/experiment.py':'old'}},{})):
                with self.assertRaisesRegex(ValueError,'before changing'):
                    control.propose(session,session/'hypothesis.json')

    def test_worker_python_guard_blocks_network_and_validation_read(self):
        script = '''
import socket, tempfile
from pathlib import Path
from sa0.worker import install_audit
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train = root/'train'; train.mkdir()
    validation = root/'validation'; validation.mkdir()
    label = validation/'label.txt'; label.write_text('hidden')
    install_audit([train])
    for action in [lambda: socket.socket(), lambda: label.read_text()]:
        try:
            action()
        except PermissionError:
            pass
        else:
            raise AssertionError('guard did not reject access')
    print('both blocked')
'''
        # Avoid temporary-directory cleanup after installing this process-global guard.
        script = script.replace("    print('both blocked')", "    print('both blocked', flush=True)\n    import os\n    os._exit(0)")
        completed = subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=30)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertIn('both blocked',completed.stdout)

    def test_rejected_candidate_restores_code_and_keeps_git_history(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory); workspace = session/'workspace'
            (workspace/'train').mkdir(parents=True)
            for name in ['mavi.py','experiment.py']:
                (workspace/'train'/name).write_text('baseline\n')
            control.git(workspace,'init','-q')
            baseline_commit = control.commit(workspace,'baseline')
            baseline_files = control.files(workspace)
            (workspace/'train/experiment.py').write_text('candidate\n')
            candidate_commit = control.commit(workspace,'candidate')
            for name, head, mae in [('baseline',baseline_commit,0.2),('candidate',candidate_commit,0.3)]:
                folder = session/'trials'/name; folder.mkdir(parents=True)
                control.write_new(folder/'result.json', {'role':name,'commit':head,
                    'evaluation':{'metrics':{'mae_float':mae,'nrms_official':0.4,'ssim_official':0.5}},
                    'train':{'parameter_count':10,'peak_allocated_mib':10}})
            control.write_new(session/'trials/candidate/source.json', {'file_sha256':control.files(workspace)})
            with patch.object(control,'load_session',return_value=({'initial_workspace_sha256':baseline_files,'baseline_commit':baseline_commit},{})), contextlib.redirect_stdout(io.StringIO()):
                control.decide(session)
            self.assertEqual(control.read(session/'decision.json')['decision'],'rollback_to_baseline')
            self.assertEqual(control.files(workspace),baseline_files)
            self.assertEqual(control.git(workspace,'show',f'{candidate_commit}:train/experiment.py'),'candidate')
            self.assertNotEqual(control.git(workspace,'rev-parse','HEAD'),candidate_commit)


if __name__ == '__main__':
    unittest.main()
