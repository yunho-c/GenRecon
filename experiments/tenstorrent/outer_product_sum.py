"""FP32 precision diagnostic using outer products and balanced additions.

This deliberately avoids a matrix reduction. Native lowering, numerical error
and cost must be measured; this is not a fast attention implementation.
"""
import torch
from torch import nn


def outer_product_sum(a,b):
    if a.dtype!=torch.float32 or b.dtype!=torch.float32:
        raise ValueError('Outer-product diagnostic requires FP32 inputs')
    if a.ndim<2 or b.ndim<2 or a.shape[-1]!=b.shape[-2] or a.shape[-1]<1:
        raise ValueError('Expected matching nonempty matrix reduction dimensions')

    def tree(begin,end):
        if end-begin==1:
            return a[...,begin:begin+1] * b[...,begin:begin+1,:]
        middle=(begin+end)//2
        # Postorder emission allows completed subtrees to release storage.
        return tree(begin,middle)+tree(middle,end)

    return tree(0,a.shape[-1])


class OuterProductSum(nn.Module):
    def forward(self,a,b):
        return outer_product_sum(a,b)
