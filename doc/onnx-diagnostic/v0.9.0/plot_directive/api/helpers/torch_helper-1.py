import torch
from onnx_diagnostic.helpers.torch_helper import study_discrepancies

t1 = torch.randn((512, 1024)) * 10
t2 = t1 + torch.randn((512, 1024))
study_discrepancies(t1, t2, title="Random noise")