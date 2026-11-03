"""Restore exact logical FP32 tensor values from the result101 base and XOR delta.

Tensor values/dtypes/shapes are preserved. Serialized storage strides and zip
container bytes are not the identity used by this compact evidence format.
"""
import argparse
import gzip
import hashlib
import io
from pathlib import Path


def tensor_digest(value):
    import torch
    value=value.detach().cpu().contiguous()
    digest=hashlib.sha256(f'{value.dtype}:{list(value.shape)}'.encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def restore(base,delta,expected,path=()):
    import torch
    if isinstance(base,torch.Tensor):
        value=base.transpose(-2,-1) if delta['transpose_last_two'] else base
        value=(value.contiguous().view(torch.int32)^delta['xor']).view(torch.float32)
        if tensor_digest(value)!=expected[repr(path)]:
            raise ValueError('Restored tensor differs: '+repr(path))
        return value
    if isinstance(base,dict):
        if base.keys()!=delta.keys():raise ValueError('Dictionary structure changed')
        return {key:restore(base[key],delta[key],expected,path+(key,)) for key in base}
    if isinstance(base,tuple):
        if len(base)!=len(delta):raise ValueError('Tuple structure changed')
        return tuple(restore(a,b,expected,path+(i,)) for i,(a,b) in enumerate(zip(base,delta)))
    raise TypeError(type(base))


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-gz',type=Path,required=True)
    parser.add_argument('--delta-gz',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    base_bytes=gzip.decompress(args.base_gz.read_bytes())
    delta=torch.load(io.BytesIO(gzip.decompress(args.delta_gz.read_bytes())),weights_only=True)
    if hashlib.sha256(base_bytes).hexdigest()!=delta['base_file_sha256']:
        raise ValueError('Base artifact changed')
    base=torch.load(io.BytesIO(base_bytes),weights_only=True)
    value=restore(base,delta['tree'],delta['target_tensor_digests'])
    torch.save(value,args.output)


if __name__=='__main__':main()
