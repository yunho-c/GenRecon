"""Keep the original Euler/CFG math on CPU while timing tensor-only forwards."""
import time

import torch

from genrecon.pipelines.samplers import FlowEulerGuidanceIntervalSampler


def features(value):
    return value if isinstance(value, torch.Tensor) else value.feats


class TensorFlowAdapter:
    """Adapt a tensor callable to the original dense/sparse sampler contract.

    Conditions remain explicit dictionary objects. The supplied callable owns
    any device transfers/caching. Texture concatenation uses normalized shape
    features, exactly as the native flow model does.
    """
    def __init__(self, forward):
        self.forward = forward
        self.calls = []

    def __call__(self, x, t, cond, concat_cond=None):
        start = time.perf_counter()
        values = features(x)
        if concat_cond is not None:
            if not torch.equal(x.coords, concat_cond.coords):
                raise ValueError("texture conditioning must use the same coordinate order")
            values = torch.cat([values, features(concat_cond)], dim=-1)
        output = self.forward(values, t, cond)
        self.calls.append({"seconds": time.perf_counter() - start, "t": t.tolist()})
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite flow output")
        return output if isinstance(x, torch.Tensor) else x.replace(output)


class MeasuredSampler(FlowEulerGuidanceIntervalSampler):
    def __init__(self, sigma_min=1e-5, progress=None):
        super().__init__(sigma_min)
        self.steps = []
        self.progress = progress

    def sample_once(self, model, x_t, t, t_prev, cond=None, **kwargs):
        before = len(model.calls)
        start = time.perf_counter()
        result = super().sample_once(model, x_t, t, t_prev, cond, **kwargs)
        self.steps.append({"step": len(self.steps) + 1, "t": t, "t_prev": t_prev,
                           "seconds": time.perf_counter() - start,
                           "forwards": len(model.calls) - before})
        if self.progress:
            self.progress(self.steps[-1])
        return result
