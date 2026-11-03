import pytest
import torch
from experiments.tenstorrent.outer_product_sum import outer_product_sum


@pytest.mark.parametrize('inner',[1,3,12,128])
@pytest.mark.parametrize('batched',[False,True])
def test_balanced_outer_sum_matches_matmul_with_broadcast_heads(inner,batched):
    torch.set_num_threads(4)
    torch.manual_seed(65)
    a=torch.randn(*((2,1) if batched else ()),17,inner)
    b=torch.randn(*((1,3) if batched else ()),inner,9)
    expected=a.double() @ b.double()
    actual=outer_product_sum(a,b)
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual.double(),expected,rtol=2e-5,atol=4e-6)


def test_small_cancellation_survives_before_final_sum():
    a=torch.tensor([[1+2**-12,1.]])
    b=torch.tensor([[4096.],[-4096.]])
    torch.testing.assert_close(outer_product_sum(a,b),torch.ones(1,1),rtol=0,atol=0)


@pytest.mark.parametrize('left,right',[
    (torch.ones(2,3,dtype=torch.bfloat16),torch.ones(3,2)),
    (torch.ones(2,3),torch.ones(4,2)),
    (torch.ones(2,0),torch.ones(0,2)),
    (torch.ones(3),torch.ones(3,2)),
])
def test_unsupported_precision_or_dimensions_are_rejected(left,right):
    with pytest.raises(ValueError):
        outer_product_sum(left,right)
