import pytest
import torch
from experiments.tenstorrent.fp32_matmul_probe import quantize_tf32


@pytest.mark.parametrize('sign',[-1,1])
def test_tf32_counterfactual_truncation_and_even_ties(sign):
    values=sign*torch.tensor([1+2**-11,1+3*2**-11,1+2**-10])
    truncated=sign*torch.tensor([1.,1+2**-10,1+2**-10])
    rounded=sign*torch.tensor([1.,1+2**-9,1+2**-10])
    torch.testing.assert_close(quantize_tf32(values),truncated,rtol=0,atol=0)
    torch.testing.assert_close(quantize_tf32(values,True),rounded,rtol=0,atol=0)


@pytest.mark.parametrize('rounded',[False,True])
def test_bf16_representable_values_survive_tf32_counterfactual(rounded):
    torch.manual_seed(62)
    values=torch.randn(512,128).bfloat16().float()
    torch.testing.assert_close(quantize_tf32(values,rounded),values,rtol=0,atol=0)
