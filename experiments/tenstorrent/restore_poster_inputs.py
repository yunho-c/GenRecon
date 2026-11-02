"""Restore pinned public poster images/cameras for stateless scene validation.

Only the eight selected images are downloaded. The filtered camera JSON retains
all 100 published frames, preserving the original crop alignment and selection.
DINO/checkpoint access is separate; this does not generate sampled latents.
"""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError('Preserve the existing restoration report')
    evidence = Path(__file__).parent/'results/sparse-decoder-integration/poster-source.json'
    original = json.loads(evidence.read_text())
    repo, revision = original['repo'], original['revision']
    names = HfApi().list_repo_files(repo, repo_type='dataset', revision=revision)
    selected = original['selected']
    patterns = [*selected, 'poster/transforms.json', 'poster/colmap/sparse/0/images.bin',
                'poster/colmap/sparse/0/points3D.bin']
    source = args.source_dir.resolve()
    def fetch(name):
        return hf_hub_download(repo, filename=name, repo_type='dataset',
                               revision=revision, local_dir=source)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, patterns))
    transforms = source/'poster/transforms.json'
    if sha256(transforms) != original['transforms_original_sha256']:
        raise ValueError('Pinned camera source hash differs from original evidence')
    data = json.loads(transforms.read_text())
    data['frames'] = [frame for frame in data['frames']
                      if 'poster/images_2/'+Path(frame['file_path']).name in names]
    if len(data['frames']) != original['frames_available']:
        raise ValueError('Published-frame inventory differs from original evidence')
    ordered = sorted(data['frames'], key=lambda frame: frame['file_path'])
    ids = [i*(len(ordered)-1)//7 for i in range(8)]
    chosen = ['poster/images_2/'+Path(ordered[index]['file_path']).name for index in ids]
    if chosen != selected:
        raise ValueError('Restored view selection differs from original evidence')
    filtered = source/'transforms_available.json'
    filtered.write_text(json.dumps(data, indent=2)+'\n')
    report = {'status': 'restored_scene_inputs', 'repo': repo, 'revision': revision,
              'source_directory': str(source), 'script_sha256': sha256(Path(__file__)),
              'frames_available': len(data['frames']), 'selected': chosen,
              'files_sha256': {name: sha256(source/name) for name in patterns},
              'filtered_transforms_sha256': sha256(filtered),
              'scope': 'Images, cameras and sparse COLMAP points only. No DINO features, flow samples or scene-quality validation.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print('Restored scene inputs:', filtered)


if __name__ == '__main__':
    main()
