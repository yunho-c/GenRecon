"""One isolated stage of scene_pipeline.py; see that driver for scope."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--stage", choices=("dino", "ss", "ss_decode", "shape", "tex", "decode"), required=True)
p.add_argument("--fixture", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--threads", type=int, default=16)
p.add_argument("--seed", type=int, default=123)
p.add_argument("--flow-group-size", type=int, default=0)
p.add_argument("--sparse-flow-layout", choices=("native", "sequence"), default="sequence")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = p.parse_args()
    if args.stage != "decode":
        for name in ("TT_METAL_HOME", "LD_LIBRARY_PATH", "PYTHONPATH"):
            os.environ.pop(name, None)
        os.environ["PJRT_DEVICE"] = "TT"
    os.environ.update(ATTN_BACKEND="sdpa", SPARSE_ATTN_BACKEND="sdpa",
                      SPARSE_CONV_BACKEND="reference" if args.stage == "decode" else "none")
    import numpy as np
    import torch
    if args.stage != "decode":
        # TT's registration/import setup must precede torchvision as well as
        # device initialization in the shared Forge environment.
        import tt_torch
        import torch_xla
    from torch import nn
    from torch.nn import functional as F
    from experiments.tenstorrent.scene_sampling import TensorFlowAdapter, MeasuredSampler, features
    from genrecon.modules.sparse import SparseTensor

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output = args.output_dir / args.stage
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    started = time.perf_counter()
    report = {"status": "running", "stage": args.stage, "torch": torch.__version__,
              "seed": args.seed, "threads": args.threads, "input_files_sha256": {},
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                  Path(__file__), Path(__file__).with_name("scene_sampling.py"),
                  Path(__file__).with_name("grouped_flow.py"),
                  ROOT / "genrecon/modules/sparse/transformer/modulated.py",
                  Path(__file__).with_name("flow_probes.py"), ROOT / "configs/pipelines/texture.json")}}

    def phase(name):
        report.update(phase=name, elapsed_seconds=time.perf_counter() - started)
        temporary = output / "result.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "result.json")
        print(name, round(report["elapsed_seconds"], 3), flush=True)

    def read(path):
        report["input_files_sha256"][str(path)] = sha256(path)
        return torch.load(path, weights_only=True, map_location="cpu")

    def compare(actual, expected):
        a, e = actual.double().reshape(-1), expected.double().reshape(-1)
        if a.shape != e.shape:
            return {"shape_match": False}
        ac, ec = a - a.mean(), e - e.mean()
        return {"finite": bool(torch.isfinite(a).all()),
                "relative_l2": ((a-e).norm() / e.norm().clamp_min(1e-30)).item(),
                "pcc": ((ac*ec).sum() / (ac.norm()*ec.norm()).clamp_min(1e-30)).item(),
                "max_abs": (a-e).abs().max().item()}

    def timed(fn):
        start = time.perf_counter()
        value = fn()
        return value, time.perf_counter() - start

    lock_fd = None
    device = None
    def open_xla():
        nonlocal lock_fd, device
        import tt_torch  # registers the backend before XLA initialization
        import torch_xla
        import torch_xla.runtime as xr
        phase("waiting_for_device")
        lock_fd = os.open("/tmp/genrecon-tt-pjrt.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        device, report["device_open_seconds"] = timed(torch_xla.device)
        assert xr.device_type() == "TT"
        report["device_type"] = xr.device_type()
        # Do not close lock_fd: it must survive all XLA atexit teardown.
        return device

    def xla_evidence():
        import torch_xla.debug.metrics as metrics
        report["executed_fallback_ops"] = list(metrics.executed_fallback_ops())
        report["aten_counters"] = {n: metrics.counter_value(n) for n in metrics.counter_names() if n.startswith("aten::")}
        report["dynamo_stats"] = dict(torch._dynamo.utils.counters["stats"])
        (output / "xla_metrics.txt").write_text(metrics.metrics_report())
        if report["executed_fallback_ops"] or report["aten_counters"]:
            raise RuntimeError("unexpected XLA CPU fallback; inspect evidence")

    def dino():
        from experiments.tenstorrent.generate_decoder_fixture import load_views
        from genrecon.modules.image_feature_extractor import DinoV3FeatureExtractor
        provenance_path = args.fixture / "provenance.json"
        report["input_files_sha256"][str(provenance_path)] = sha256(provenance_path)
        provenance = json.loads(provenance_path.read_text())
        phase("preprocess_images")
        data, report["preprocess_seconds"] = timed(lambda: load_views(
            Path(provenance["source_transforms"]), Path(provenance["images"][0]["path"]).parent,
            len(provenance["images"]), provenance["center"], provenance["extent"]))
        images, intrinsics, extrinsics, sources = data
        assert sources == provenance["images"]
        expected = read(args.fixture / "conditioning.pt")
        torch.testing.assert_close(intrinsics, expected["intrinsics"], rtol=0, atol=0)
        torch.testing.assert_close(extrinsics, expected["extrinsics"], rtol=0, atol=0)
        phase("load_dino")
        extractor, report["model_load_seconds"] = timed(lambda: DinoV3FeatureExtractor(str(args.checkpoints / "dinov3-vitl16")))
        class Extractor(nn.Module):
            extract_features = DinoV3FeatureExtractor.extract_features
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, image):
                return self.extract_features(image)
        wrapper = Extractor(extractor.model).eval()
        wrapper.model.set_attn_implementation("eager")
        inv_freq = wrapper.model.rope_embeddings.inv_freq.clone()
        wrapper.to(torch.bfloat16)
        wrapper.model.rope_embeddings.inv_freq = inv_freq
        inputs, report["normalize_seconds"] = timed(lambda: extractor.transform(images).to(torch.bfloat16))
        open_xla()
        _, report["model_to_device_seconds"] = timed(lambda: wrapper.to(device))
        compiled = torch.compile(wrapper, backend="tt", fullgraph=True)
        def infer(i):
            return compiled(inputs[i:i+1].to(device)).cpu().float()
        phase("compile_dino")
        _, report["first_inference_seconds"] = timed(lambda: infer(0))
        phase("eight_warm_views")
        outputs, times = [], []
        start = time.perf_counter()
        for i in range(len(images)):
            value, seconds = timed(lambda: infer(i))
            outputs.append(value)
            times.append(seconds)
        all_features = torch.cat(outputs)[None]
        report.update(warm_views_seconds=time.perf_counter()-start, view_seconds=times,
                      vs_saved_cpu_fp32=compare(all_features, expected["features"]), images=sources,
                      precision="BF16, original FP32 RoPE frequencies")
        torch.save({"features": all_features, "intrinsics": intrinsics, "extrinsics": extrinsics}, output / "conditioning.pt")
        report["warm_path_seconds"] = report["preprocess_seconds"] + report["normalize_seconds"] + report["warm_views_seconds"]
        xla_evidence()

    def flow():
        from experiments.tenstorrent.flow_probes import load_model, DenseFlowProbe, NativeSparseFlowProbe, SingleChunkSLatProbe
        from experiments.tenstorrent.reference import float32_reference
        from experiments.tenstorrent.probes import SparseAggregatorProbe
        conditioning = read(args.output_dir / "dino/conditioning.pt")
        cfg = json.loads((ROOT / "configs/pipelines/texture.json").read_text())["args"]
        family = {"ss": "sparse_structure", "shape": "shape_slat", "tex": "tex_slat"}[args.stage]
        params = cfg[family + "_sampler"]["params"]
        report["sampling"] = params
        phase("load_flow")
        loaded, report["model_load_seconds"] = timed(lambda: load_model(args.stage, args.checkpoints))
        model, model_config, provenance = loaded
        provenance["input_kind"] = "real poster images; TT-generated upstream conditioning/occupancy"
        report["checkpoint"] = provenance
        coords = None if args.stage == "ss" else read(args.output_dir / "ss_decode/sparse_structure.pt")["coords"]
        report["active_voxels"] = 4096 if coords is None else len(coords)
        feats = conditioning["features"]
        extr, intr = conditioning["extrinsics"], conditioning["intrinsics"]
        phase("cpu_projection")
        if coords is None:
            projection, report["projection_seconds"] = timed(lambda: model.projection(feats, extr, intr))
            projected, mask, camera = projection
            if camera is not None:
                raise ValueError("this probe expects the configured camera-free aggregator")
            aggregator = copy.deepcopy(model.aggregator).eval()
            agg_args = (projected, mask)
        else:
            sparse = SparseTensor(torch.zeros(len(coords), 1), coords)
            projection, report["projection_seconds"] = timed(lambda: model.projection(sparse, feats, extr, intr))
            projected, mask = projection
            if not torch.equal(projected.coords, coords):
                raise ValueError("unexpected projection coordinate change")
            aggregator = SparseAggregatorProbe(1024, len(coords))
            aggregator.coords = coords
            aggregator.aggregator = copy.deepcopy(model.aggregator)
            aggregator.eval()
            agg_args = (projected.feats, mask)
        with float32_reference(aggregator):
            agg_ref, report["aggregator_cpu_reference_seconds"] = timed(lambda: aggregator(*agg_args))
        open_xla()
        aggregator.to(device)
        agg_compiled = torch.compile(aggregator, backend="tt", fullgraph=True)
        upload_start = time.perf_counter()
        agg_device_args = tuple(x.to(device, dtype=torch.bfloat16) if x.is_floating_point() else x.to(device) for x in agg_args)
        report["aggregator_input_upload_seconds"] = time.perf_counter() - upload_start
        def agg_infer():
            return agg_compiled(*agg_device_args).cpu().float()
        phase("compile_aggregator")
        aggr, report["aggregator_first_seconds"] = timed(agg_infer)
        report["aggregator_vs_cpu_fp32"] = compare(aggr, agg_ref)
        report["aggregator_warm_seconds"] = [timed(agg_infer)[1] for _ in range(3)]
        del agg_compiled, aggregator, agg_device_args, agg_args, agg_ref, projection, projected
        gc.collect()
        positive = {"cond_2D": feats[:, 0].contiguous(), "cond_3D": aggr.contiguous()}
        negative = {key: torch.zeros_like(value) for key, value in positive.items()}
        # Explicit per-stage noise is independent of checkpoint construction RNG.
        noise_seed = args.seed + {"ss": 0, "shape": 1, "tex": 2}[args.stage]
        generator = torch.Generator().manual_seed(noise_seed)
        values = torch.randn((1, 8, 16, 16, 16) if coords is None else (len(coords), 32), generator=generator)
        noise = values if coords is None else SparseTensor(values, coords)
        concat = None
        if args.stage == "tex":
            shape = read(args.output_dir / "shape/slat.pt")
            assert torch.equal(shape["coords"], coords)
            concat = SparseTensor(shape["normalized"], coords)
        reference_probe = DenseFlowProbe(model) if coords is None else NativeSparseFlowProbe(model, coords, precompute_rope=True)
        probe = (SingleChunkSLatProbe(model, model_config, coords).eval()
                 if coords is not None and args.sparse_flow_layout == "sequence" else reference_probe)
        if args.sparse_flow_layout == "sequence" and args.flow_group_size:
            raise ValueError("sequence layout and native block grouping are separate experiments")
        report["sparse_flow_layout"] = args.sparse_flow_layout if coords is not None else None
        first_values = values if concat is None else torch.cat([values, concat.feats], dim=-1)
        t = torch.tensor([1000.])
        phase("cpu_first_forward_reference")
        with float32_reference(nn.ModuleList([reference_probe, probe])):
            reference, report["cpu_forward_seconds"] = timed(lambda: reference_probe(first_values, t, positive["cond_2D"], positive["cond_3D"]))
            if probe is not reference_probe:
                equivalent, report["adapter_cpu_equivalence_seconds"] = timed(lambda: probe(first_values, t, positive["cond_2D"], positive["cond_3D"]))
                report["adapter_vs_native_cpu"] = compare(equivalent, reference)
                torch.testing.assert_close(equivalent, reference)
        group_size = args.flow_group_size if coords is not None else 0
        if group_size:
            from experiments.tenstorrent.grouped_flow import flow_groups
            groups = flow_groups(model, coords, group_size)
            _, report["model_to_device_seconds"] = timed(lambda: groups.to(device))
            compiled_groups = [torch.compile(group, backend="tt", fullgraph=True) for group in groups]
        else:
            _, report["model_to_device_seconds"] = timed(lambda: probe.to(device))
            compiled = torch.compile(probe, backend="tt", fullgraph=True)
        report["flow_group_size"] = group_size
        report["compilation_boundary"] = ("native groups with synchronized CPU feature roundtrips" if group_size else
            "whole active-token sequence" if probe is not reference_probe else "whole native flow")
        start = time.perf_counter()
        conditions = {id(cond): tuple(cond[key].to(device) for key in ("cond_2D", "cond_3D")) for cond in (positive, negative)}
        report["condition_upload_seconds"] = time.perf_counter() - start
        def forward(x, timestep, cond):
            time_device = timestep.to(device)
            if group_size:
                h = x
                for group in compiled_groups:
                    # Bound liveness; BF16 intermediate features stay BF16.
                    h = group(h.to(device), time_device, *conditions[id(cond)]).cpu()
                return h.float()
            return compiled(x.to(device), time_device, *conditions[id(cond)]).cpu().float()
        phase("compile_flow")
        first, report["first_forward_seconds"] = timed(lambda: forward(first_values, t, positive))
        report["first_forward_vs_cpu_fp32"] = compare(first, reference)
        phase("warm_flow")
        # Exercise both conditions before measuring the full sampler.
        _, report["negative_warmup_seconds"] = timed(lambda: forward(first_values, t, negative))
        report["warm_forward_seconds"] = [timed(lambda: forward(first_values, t, positive))[1] for _ in range(3)]
        adapter = TensorFlowAdapter(forward)
        def progress(step):
            print("step", step, flush=True)
        sampler = MeasuredSampler(progress=progress)
        report["dynamo_before_sampling"] = dict(torch._dynamo.utils.counters["stats"])
        phase("sample_12_steps")
        result, report["sampler_seconds"] = timed(lambda: sampler.sample(adapter, noise,
            cond=positive, neg_cond=negative, concat_cond=concat, verbose=False, **params))
        report.update(steps=sampler.steps, forwards=adapter.calls, forward_count=len(adapter.calls),
                      noise_seed=noise_seed, dynamo_after_sampling=dict(torch._dynamo.utils.counters["stats"]))
        assert len(adapter.calls) == {"ss": 22, "shape": 21, "tex": 18}[args.stage]
        final = features(result.samples)
        if not torch.isfinite(final).all():
            raise ValueError("nonfinite sampled latent")
        torch.save({"noise": values, "noise_seed": noise_seed}, output / "noise.pt")
        if coords is None:
            torch.save({"latent": final}, output / "latent.pt")
        else:
            assert torch.equal(result.samples.coords, coords)
            norm = cfg[family + "_normalization"]
            denormalized = final * torch.tensor(norm["std"])[None] + torch.tensor(norm["mean"])[None]
            torch.save({"coords": coords, "normalized": final, "feats": denormalized}, output / "slat.pt")
        report["warm_path_seconds"] = (report["projection_seconds"] + statistics.median(report["aggregator_warm_seconds"])
            + report["aggregator_input_upload_seconds"] + report["condition_upload_seconds"] + report["sampler_seconds"])
        report["sampler_host_and_instrumentation_seconds"] = report["sampler_seconds"] - sum(c["seconds"] for c in adapter.calls)
        xla_evidence()

    def ss_decode():
        from genrecon import models
        latent = read(args.output_dir / "ss/latent.pt")["latent"]
        path = args.checkpoints / "trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        report["checkpoint_sha256"] = {suffix: sha256(path.with_suffix(suffix)) for suffix in (".json", ".safetensors")}
        phase("load_ss_decoder")
        model, report["model_load_seconds"] = timed(lambda: models.from_pretrained(str(path)).float().eval())
        model.convert_to_fp32()
        reference, report["cpu_reference_seconds"] = timed(lambda: model(latent))
        open_xla()
        model.to(device)
        compiled = torch.compile(model, backend="tt", fullgraph=True)
        def infer():
            return compiled(latent.to(device)).cpu().float()
        phase("compile_ss_decoder")
        actual, report["first_inference_seconds"] = timed(infer)
        report["warm_seconds"] = [timed(infer)[1] for _ in range(3)]
        report["vs_cpu_fp32"] = compare(actual, reference)
        def occupancy(logits):
            return torch.argwhere(F.max_pool3d((logits > 0).float(), 2, 2) > 0)[:, [0, 2, 3, 4]].int()
        coords, report["occupancy_seconds"] = timed(lambda: occupancy(actual))
        expected = occupancy(reference)
        report.update(active_voxels=len(coords), cpu_reference_voxels=len(expected), coordinates_equal=torch.equal(coords, expected),
                      logit_sign_disagreements=int(((actual > 0) != (reference > 0)).sum()))
        if not len(coords):
            raise ValueError("TT occupancy produced no active voxels")
        torch.save({"latent": latent, "coords": coords, "logits": actual}, output / "sparse_structure.pt")
        report["warm_path_seconds"] = statistics.median(report["warm_seconds"]) + report["occupancy_seconds"]
        xla_evidence()

    def decode():
        from experiments.tenstorrent.sparse_decoder_probe import load_model, save_sparse, save_guides
        from genrecon.modules.sparse import config
        from genrecon.backends.tenstorrent.sparse_decoder import TopologyCache
        import ttnn
        config.set_conv_backend("reference")
        shape_input = read(args.output_dir / "shape/slat.pt")
        tex_input = read(args.output_dir / "tex/slat.pt")
        assert torch.equal(shape_input["coords"], tex_input["coords"])
        np.savez_compressed(output / "decoder_input.npz", coords=shape_input["coords"].numpy(),
            shape_feats=shape_input["feats"].numpy(), texture_feats=tex_input["feats"].numpy())
        xshape = SparseTensor(shape_input["feats"], shape_input["coords"])
        xtex = SparseTensor(tex_input["feats"], tex_input["coords"])
        phase("waiting_for_device")
        nonlocal lock_fd, device
        lock_fd = os.open("/tmp/genrecon-tt-pjrt.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        device, report["device_open_seconds"] = timed(lambda: ttnn.open_device(device_id=0))
        cache = TopologyCache()
        report["precision"] = "bf16"
        report["warm_path_seconds"] = 0.
        guides = None
        shape_coords = None
        try:
            for name, stem, values in (("shape", "shape_dec_next_dc_f16c32_fp16", xshape),
                                       ("texture", "tex_dec_next_dc_f16c32_fp16", xtex)):
                phase("load_" + name)
                path = args.checkpoints / "trellis2/ckpts" / stem
                data = report[name] = {"checkpoint_sha256": sha256(path.with_suffix(".safetensors"))}
                model, data["model_load_seconds"] = timed(lambda: load_model(path))
                model.enable_tenstorrent(device, precision="bf16", row_chunk=1024, topology_cache=cache)
                data["prepare_seconds"] = model._tenstorrent_backend.prepare_seconds
                def infer():
                    return model.forward_features(values, return_subs=True) if name == "shape" else model(values, guide_subs=guides)
                phase("first_" + name)
                first, data["first_seconds"] = timed(infer)
                actual = first[0] if name == "shape" else first
                if name == "shape":
                    guides = first[1]
                    shape_coords = actual.coords
                    save_guides(output / "guides.pt", guides)
                else:
                    report["geometry_material_coordinates_equal"] = torch.equal(actual.coords, shape_coords)
                    if not report["geometry_material_coordinates_equal"]:
                        raise ValueError("shape and texture decoder coordinate orders differ")
                if not torch.isfinite(actual.feats).all():
                    raise ValueError("nonfinite decoder output")
                data.update(output_voxels=len(actual.coords), warm_seconds=[])
                phase("warm_" + name)
                for _ in range(3):
                    repeat, seconds = timed(infer)
                    repeated = repeat[0] if name == "shape" else repeat
                    assert torch.equal(actual.coords, repeated.coords) and torch.equal(actual.feats, repeated.feats)
                    data["warm_seconds"].append(seconds)
                data["warm_median_seconds"] = statistics.median(data["warm_seconds"])
                report["warm_path_seconds"] += data["warm_median_seconds"]
                save_sparse(output / (name + ".pt"), actual)
                model.disable_tenstorrent()
                cache.clear()
                del model, repeat, repeated, actual, first
                gc.collect()
        finally:
            cache.clear()
            ttnn.close_device(device)

    phase("initialize")
    try:
        with torch.no_grad():
            if args.stage == "dino":
                dino()
            elif args.stage in ("ss", "shape", "tex"):
                flow()
            elif args.stage == "ss_decode":
                ss_decode()
            else:
                decode()
        report["status"] = "completed"
        phase("complete")
    except Exception as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        phase(report.get("phase", "initialize"))
        raise


if __name__ == "__main__":
    main()
