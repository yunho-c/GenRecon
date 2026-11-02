import pytest
import torch
from experiments.tenstorrent.precision_reference import PipelinePrecisionMode


@pytest.mark.parametrize('operation', ['linear','matmul','attention'])
def test_small_native_bf16_oracle(operation):
    torch.set_num_threads(4)
    torch.manual_seed(31)
    if operation=='linear':
        x,w,b=[torch.randn(*shape).bfloat16() for shape in ((17,48),(96,48),(96,))]
        fn=lambda:torch.nn.functional.linear(x,w,b)
    elif operation=='matmul':
        x,w=[torch.randn(*shape).bfloat16() for shape in ((2,4,17,12),(2,4,12,19))]
        fn=lambda:x@w
    else:
        q,k,v=[torch.randn(2,4,17,12).bfloat16() for _ in range(3)]
        fn=lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v)
    expected=fn()
    with PipelinePrecisionMode() as mode:
        actual=fn()
    assert actual.dtype==expected.dtype==torch.bfloat16
    torch.testing.assert_close(actual.float(),expected.float(),rtol=.008,atol=.002)
    assert sum(mode.promoted_ops.values())>0


def test_fp32_unchanged_and_exception_restores_dispatch():
    x=torch.randn(3,7)
    expected=x@x.T
    with pytest.raises(RuntimeError,match='test exit'):
        with PipelinePrecisionMode() as mode:
            torch.testing.assert_close(x@x.T,expected,rtol=0,atol=0)
            assert not mode.promoted_ops
            raise RuntimeError('test exit')
    torch.testing.assert_close(x@x.T,expected,rtol=0,atol=0)
