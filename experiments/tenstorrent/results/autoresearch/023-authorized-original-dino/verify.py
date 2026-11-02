from pathlib import Path
import json,sys
sys.path.insert(0,'/home/user/GitHub/GenRecon')
import torch
from transformers import DINOv3ViTConfig
from genrecon.modules.image_feature_extractor import DinoV3FeatureExtractor
from experiments.tenstorrent.generate_decoder_fixture import load_views
from experiments.tenstorrent.experiment_identity import sha256
from experiments.tenstorrent.trajectory import tensor_digest
from experiments.tenstorrent.validation import compare_tensors

torch.set_num_threads(16)
original=Path('/home/user/models/genrecon/dinov3-facebook-authorized')
converted=Path('/home/user/models/genrecon/dinov3-timm-validated')
fixture=Path('/home/user/tt-scratchpad/genrecon/poster-timm-conditioning')
report=json.loads((original/'restoration.json').read_text())
report['script_sha256']=sha256(Path(__file__))
report['weights_equal']=report['files_sha256']['model.safetensors']==sha256(converted/'model.safetensors')
assert report['weights_equal']
a=DINOv3ViTConfig.from_pretrained(original).to_dict()
b=DINOv3ViTConfig.from_pretrained(converted).to_dict()
ignored=('_name_or_path','transformers_version')
for config in (a,b):
 for key in ignored: config.pop(key,None)
report['effective_configs_equal']=a==b
report['effective_config_differences']={key:[a.get(key),b.get(key)] for key in a.keys()|b.keys() if a.get(key)!=b.get(key)}
assert report['effective_configs_equal'],report['effective_config_differences']
p=json.loads((fixture/'provenance.json').read_text())
images,intr,extr,sources=load_views(Path(p['source_transforms']),Path(p['images'][0]['path']).parent,len(p['images']),p['center'],p['extent'])
assert sources==p['images']
expected=torch.load(fixture/'conditioning.pt',weights_only=True)
extractor=DinoV3FeatureExtractor(str(original))
with torch.no_grad(): actual=torch.cat([extractor.extract_features(extractor.transform(image[None])) for image in images])[None]
report['features_vs_existing_fixture']=compare_tensors(actual,expected['features'],max_relative_l2=1e-6,min_pcc=.999999)
report['features_bitwise_equal']=torch.equal(actual,expected['features'])
report['cameras_bitwise_equal']=torch.equal(intr,expected['intrinsics']) and torch.equal(extr,expected['extrinsics'])
assert report['features_bitwise_equal'] and report['cameras_bitwise_equal']
report['feature_tensor_sha256']=tensor_digest(actual)
report['existing_conditioning_sha256']=sha256(fixture/'conditioning.pt')
report['status']='authorized_original_matches_restored_weights_config_and_features'
report['scope']='Canonical pinned original source downloaded using user-authorized HF login. All effective config fields match converted checkpoint; features for all eight original cropped views and camera tensors are bitwise equal to the already-running matched sampler fixture. No need to replace those inputs.'
Path('/home/user/tt-scratchpad/genrecon/dino-conversion-check/authorized-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
