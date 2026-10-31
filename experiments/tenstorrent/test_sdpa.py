"""Check ragged sample isolation and all supported SDPA calling conventions."""
import os
os.environ["SPARSE_CONV_BACKEND"] = "none"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
os.environ["ATTN_BACKEND"] = "sdpa"

import unittest
from fractions import Fraction
import torch
from genrecon.modules.sparse.basic import VarLenTensor, SparseTensor
from genrecon.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention as attention


def reference(q, k, v):
    weights = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) / q.shape[-1]**0.5, dim=-1)
    return torch.einsum("hqk,khd->qhd", weights, v)


class TestSDPA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.q = [torch.randn(n, 2, 8) for n in (3, 5)]
        self.k = [torch.randn(n, 2, 8) for n in (4, 7)]
        self.v = [torch.randn(n, 2, 6) for n in (4, 7)]

    def test_ragged_self_packed(self):
        value = VarLenTensor.from_tensor_list([torch.stack([q,q,q], dim=1) for q in self.q])
        result = attention(value)
        for expected, actual in zip(self.q, result.to_tensor_list()):
            torch.testing.assert_close(actual, reference(expected, expected, expected))

    def test_ragged_cross_separate(self):
        result = attention(*(VarLenTensor.from_tensor_list(t) for t in (self.q,self.k,self.v)))
        for q,k,v,out in zip(self.q,self.k,self.v,result.to_tensor_list()):
            torch.testing.assert_close(out, reference(q,k,v))

    def test_ragged_q_dense_kv_packed(self):
        kv = torch.randn(2, 4, 2, 2, 8)
        result = attention(VarLenTensor.from_tensor_list(self.q), kv)
        for i,out in enumerate(result.to_tensor_list()):
            torch.testing.assert_close(out, reference(self.q[i],kv[i,:,0],kv[i,:,1]))

    def test_dense_q_ragged_kv(self):
        q = torch.randn(2, 3, 2, 8)
        result = attention(q, VarLenTensor.from_tensor_list(self.k), VarLenTensor.from_tensor_list(self.v))
        for i in range(2):
            torch.testing.assert_close(result[i], reference(q[i], self.k[i], self.v[i]))

    def test_spatial_cache_separates_resolutions(self):
        tensor = SparseTensor(torch.ones(2,8), torch.zeros(2,4,dtype=torch.int32), shape=torch.Size([1,8]))
        tensor.register_spatial_cache("neighbors", "original")
        changed = tensor.replace(torch.zeros(2,8))
        changed._scale = (Fraction(1,2),)*3
        self.assertIsNone(changed.get_spatial_cache("neighbors"))
        changed.register_spatial_cache("neighbors", "upsampled")
        self.assertEqual(tensor.get_spatial_cache("neighbors"), "original")
        changed._scale = (1,1,1)
        self.assertEqual(changed.get_spatial_cache("neighbors"), "original")


if __name__ == "__main__":
    unittest.main()
