import unittest
import torch
from train.mavi import MAVI
class TestMAVIContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MAVI()
        self.input = torch.randn(1,1,24,32,32)
        
    def test_forward_shape_and_finiteness(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.input)
        self.assertEqual(pred.shape,(1,32,32))
        self.assertTrue(torch.isfinite(pred).all().item())
        
    def test_backward_gradient_are_finite(self):
        self.model.train()
        target = torch.randn(1,32,32)
        pred = self.model(self.input)
        loss = torch.nn.functional.l1_loss(pred,target)
        loss.backward()
        gradients = [parameter
                     for parameter in self.model.parameters()
                     if parameter.requires_grad and parameter.grad is not None]
        self.assertGreater(len(gradients),0)
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients),"Some trainable parameters did not receive gradients.",)
        self.assertTrue(all(torch.isfinite(parameter.grad).all().item()
                            for parameter in gradients),
                         "Some gradients contain NaN or Inf.",)
if __name__ == "__main__":
    unittest.main()