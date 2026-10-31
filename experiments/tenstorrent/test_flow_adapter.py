"""Check active-token adapters against native sparse networks, including LoRA."""
import os
os.environ["SPARSE_CONV_BACKEND"] = "none"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
os.environ["ATTN_BACKEND"] = "sdpa"

import unittest
import torch
from easydict import EasyDict
from genrecon.utils.model_wrapper_utils import build_single_model
from genrecon.modules.utils import convert_module_to
from experiments.tenstorrent.flow_probes import NativeSparseFlowProbe, SingleChunkSLatProbe
from experiments.tenstorrent.reference import float32_reference


class TestFlowAdapter(unittest.TestCase):
    def test_shape_and_preconcatenated_texture(self):
        torch.set_num_threads(4)
        for channels in (32,64):
            with self.subTest(channels=channels), torch.no_grad():
                torch.manual_seed(7)
                cfg = EasyDict(name="SLatFlowModel", args=dict(resolution=8,in_channels=channels,
                    out_channels=32,model_channels=96,cond_channels=32,num_blocks=3,num_heads=4,
                    mlp_ratio=2,pe_mode="rope",share_mod=True,qk_rms_norm=True,qk_rms_norm_cross=True),
                    lora=dict(r=4,alpha=8),projection=dict(image_size=512))
                model = build_single_model(cfg).eval()
                for name,param in model.named_parameters():
                    if "lora_B" in name or name.startswith(("proj_linears.","out_layer.")):
                        torch.nn.init.normal_(param,std=.02)
                model.convert_to(torch.bfloat16)
                model.proj_linears.apply(lambda m:convert_module_to(m,torch.bfloat16))
                coords = torch.tensor([[0,1,2,0],[0,3,4,7],[0,0,1,5],[0,7,7,7]],dtype=torch.int32)
                native = NativeSparseFlowProbe(model,coords)
                adapted = SingleChunkSLatProbe(model,cfg,coords).eval()
                inputs=(torch.randn(4,channels),torch.tensor([600.]),torch.randn(1,9,32),torch.randn(4,32))
                expected,actual=native(*inputs),adapted(*inputs)
                self.assertGreater(expected.std().item(),.01)
                torch.testing.assert_close(actual,expected)
                snapshots = {name:t.clone() for name,t in adapted.state_dict().items()}
                with float32_reference(adapted):
                    self.assertEqual(adapted.dtype,torch.float32)
                    self.assertTrue(adapted.phases.is_complex())
                    self.assertTrue(torch.isfinite(adapted(*inputs)).all())
                self.assertEqual(adapted.dtype,torch.bfloat16)
                for name,tensor in adapted.state_dict().items():
                    self.assertEqual(tensor.dtype,snapshots[name].dtype)
                    torch.testing.assert_close(tensor,snapshots[name],rtol=0,atol=0)
                torch.testing.assert_close(adapted(*inputs),expected)


if __name__ == "__main__":
    unittest.main()
