"""Restore unmasked DINOv3 ViT-L16 inference from the pinned public timm release.

The tool validates unmasked features and checks whether serialization reproduces
the historical Facebook weights-file hash. The omitted mask token is restored
as zero; exact file-hash agreement verifies that value as well as learned weights.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import traceback

REPO = 'timm/vit_large_patch16_dinov3.lvd1689m'
REVISION = '30c1109559f65dea34316b0d4842d35c5771fe11'
WEIGHTS_SHA256 = '45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'
HISTORICAL_SHA256 = 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179'
CAT_SHA256 = 'cf6f3c4befa148732c7453e0de5afab00f682427435fead2d88b07a9615cdac2'
CAT_URL = 'https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000039769.png'
CONVERSION_URL = 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def convert_state(state):
    import torch
    converted = {}
    for key, value in state.items():
        if key == 'cls_token':
            name = 'embeddings.cls_token'
        elif key == 'reg_token':
            name = 'embeddings.register_tokens'
        elif key.startswith('patch_embed.proj.'):
            name = key.replace('patch_embed.proj.', 'embeddings.patch_embeddings.')
        elif key.startswith('norm.'):
            name = key
        else:
            match = re.fullmatch(r'blocks\.(\d+)\.(.+)', key)
            if match is None:
                raise ValueError(f'Unmapped timm parameter: {key}')
            index, part = match.groups()
            prefix = 'model.layer.' + index + '.'
            if part == 'attn.qkv.weight':
                for tag, tensor in zip(('q', 'k', 'v'), value.chunk(3, dim=0), strict=True):
                    converted[prefix + 'attention.' + tag + '_proj.weight'] = tensor.contiguous()
                    if tag != 'k':
                        # Public timm release removes original all-zero QKV biases.
                        converted[prefix + 'attention.' + tag + '_proj.bias'] = value.new_zeros(value.shape[1])
                continue
            if part.startswith('attn.proj.'):
                name = prefix + part.replace('attn.proj.', 'attention.o_proj.')
            elif part.startswith('mlp.fc1.'):
                name = prefix + part.replace('mlp.fc1.', 'mlp.up_proj.')
            elif part.startswith('mlp.fc2.'):
                name = prefix + part.replace('mlp.fc2.', 'mlp.down_proj.')
            elif part in ('gamma_1', 'gamma_2'):
                name = prefix + 'layer_scale' + part[-1] + '.lambda1'
            elif part.startswith(('norm1.', 'norm2.')):
                name = prefix + part
            else:
                raise ValueError(f'Unmapped timm parameter: {key}')
        if name in converted:
            raise ValueError(f'Duplicate destination parameter: {name}')
        converted[name] = value
    converted['embeddings.mask_token'] = torch.zeros(1, 1, 1024)
    return converted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--reference-image', type=Path, required=True, help=CAT_URL)
    p.add_argument('--validation-images', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=16)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'repo': REPO, 'revision': REVISION,
              'source_card': 'https://huggingface.co/' + REPO,
              'official_conversion_source': CONVERSION_URL,
              'script_sha256': sha256(__file__),
              'scope': 'Unmasked inference conversion; original gated HF checkpoint hash is not verified. Masked-image use is unsupported. Default FP32 RoPE periods match Transformers; no BF16-period substitution.',
              'restored_parameters': 'Zero query/value biases and an unused zero mask-token placeholder. All public timm learned tensors mapped; strict loading required.'}

    def save():
        (args.output_dir / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        import torch
        import timm
        from huggingface_hub import hf_hub_download
        from PIL import Image
        from safetensors.torch import load_file
        from torchvision import transforms
        from torchvision.transforms import functional as TF
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        from experiments.tenstorrent.validation import compare_tensors
        from genrecon.modules.image_feature_extractor import DinoV3FeatureExtractor
        torch.set_num_threads(args.threads)
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ('torch', 'torchvision', 'transformers', 'timm', 'safetensors')}
        for name in ('config.json', 'README.md', 'LICENSE.md', 'model.safetensors'):
            hf_hub_download(REPO, filename=name, revision=REVISION, local_dir=args.source_dir)
        weights = args.source_dir / 'model.safetensors'
        report['source_sha256'] = {name: sha256(args.source_dir / name)
                                  for name in ('config.json', 'README.md', 'LICENSE.md', 'model.safetensors')}
        if report['source_sha256']['model.safetensors'] != WEIGHTS_SHA256:
            raise ValueError('Pinned public weights hash mismatch')
        if sha256(args.reference_image) != CAT_SHA256:
            raise ValueError('Reference-image hash mismatch')
        state = load_file(str(weights))
        config = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096,
            num_hidden_layers=24, num_attention_heads=16, num_register_tokens=4)
        model = DINOv3ViTModel(config).eval()
        model.load_state_dict(convert_state(state), strict=True)
        report['strict_loading'] = True
        normalize = transforms.Normalize((.485, .456, .406), (.229, .224, .225))

        def pixels(path, size):
            return normalize(TF.resize(TF.to_tensor(Image.open(path).convert('RGB')),
                                       [size, size], antialias=True)[None])

        with torch.no_grad():
            output = model(pixels(args.reference_image, 224))
        actual = {'cls': output.pooler_output[0, :5].tolist(),
                  'patch': output.last_hidden_state[0, 5, :5].tolist()}
        expected = {'cls': [.484527, -.582214, .480636, .592040, .945166],
                    'patch': [-.211367, -.490863, -.257131, .101763, .154511]}
        report['official_smoke'] = {'actual': actual, 'expected': expected,
                                    'atol': 1e-3, 'rtol': 1e-3, 'reference_image_sha256': CAT_SHA256}
        save()
        for key in expected:
            torch.testing.assert_close(torch.tensor(actual[key]), torch.tensor(expected[key]),
                                       atol=1e-3, rtol=1e-3)
        original = timm.create_model('vit_large_patch16_dinov3', pretrained=False).eval()
        original.load_state_dict(state, strict=True)
        original.norm = torch.nn.Identity()  # GenRecon uses non-affine final normalization.

        class Extractor(torch.nn.Module):
            extract_features = DinoV3FeatureExtractor.extract_features
            def __init__(self, model):
                super().__init__()
                self.model = model

        extractor = Extractor(model)
        report['feature_checks'] = []
        for path in args.validation_images:
            with torch.no_grad():
                image = pixels(path, 512)
                converted = extractor.extract_features(image)
                public = torch.nn.functional.layer_norm(original.forward_features(image), (1024,))
            check = compare_tensors(converted, public, max_relative_l2=1e-4, min_pcc=.999999)
            report['feature_checks'].append({'image': str(path), 'image_sha256': sha256(path),
                                             'shape': list(converted.shape), 'check': check})
            save()
            if not check['passed']:
                raise ValueError(f'Public timm feature comparison failed: {path}')
        model.save_pretrained(args.output_dir)
        report['converted_sha256'] = {name: sha256(args.output_dir / name)
                                      for name in ('config.json', 'model.safetensors')}
        report['historical_expected_sha256'] = HISTORICAL_SHA256
        report['historical_evidence'] = 'results/sparse-decoder-integration/poster-cpu-fixture.json:dino_sha256'
        report['historical_file_hash_reproduced'] = (
            report['converted_sha256']['model.safetensors'] == HISTORICAL_SHA256)
        if report['historical_file_hash_reproduced']:
            report['scope'] = 'Unmasked GenRecon features validated; serialized weights exactly match the historical Facebook checkpoint SHA256, including restored zero biases and mask token. Config follows official Transformers conversion, not a historical config-file hash. Current software versions still require fresh paired scene validation.'
        report['status'] = 'validated_unmasked_conversion'
        save()
        print(report['status'], args.output_dir, flush=True)
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    # Support direct invocation as well as python -m.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
