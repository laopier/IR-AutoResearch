from pathlib import Path
import tempfile
import unittest

from sa0.agent_loop import validate_edits, apply_edits


SOURCE = '''import torch
def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-4)
def compute_loss(prediction, target):
    return (prediction-target).abs().mean()
def train_batch(model, optimizer, feature, target):
    return 1.0
'''


class AgentEditBoundaryTests(unittest.TestCase):
    def proposal(self, edits):
        return dict(hypothesis='test',change='test',expected_effect='test',risk='test',
                    controlled_variables=['seed'],plan=['edit'],edits=edits)

    def test_rejects_path_traversal_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'train').mkdir(); path=root/'train/experiment.py'; path.write_text(SOURCE)
            proposal=self.proposal([dict(path='../outside.py',old='lr=2e-4',new='lr=1e-4')])
            with self.assertRaises(ValueError): apply_edits(root,proposal)
            self.assertEqual(path.read_text(),SOURCE)

    def test_bad_second_edit_does_not_partially_apply_first(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'train').mkdir(); path=root/'train/experiment.py'; path.write_text(SOURCE)
            proposal=self.proposal([dict(path='train/experiment.py',old='lr=2e-4',new='lr=1e-4'),
                                   dict(path='train/experiment.py',old='missing snippet',new='replacement')])
            with self.assertRaises(ValueError): apply_edits(root,proposal)
            self.assertEqual(path.read_text(),SOURCE)

    def test_rejects_training_budget_and_module_side_effect_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'train').mkdir(); (root/'train/experiment.py').write_text(SOURCE)
            for old,new in [('return 1.0','return 2.0'),('import torch','import torch\nprint("side effect")')]:
                with self.subTest(old=old), self.assertRaises(ValueError):
                    validate_edits(root,self.proposal([dict(path='train/experiment.py',old=old,new=new)]))

    def test_valid_optimizer_change_applies(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'train').mkdir(); path=root/'train/experiment.py'; path.write_text(SOURCE)
            apply_edits(root,self.proposal([dict(path='train/experiment.py',old='lr=2e-4',new='lr=1e-4')]))
            self.assertEqual(path.read_text(),SOURCE.replace('lr=2e-4','lr=1e-4'))


if __name__=='__main__': unittest.main()
