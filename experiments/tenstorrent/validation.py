"""Numerical smoke gates for experiments; not scene-quality acceptance."""
import torch


def compare_tensors(actual, expected, *, max_relative_l2=0.1, min_pcc=0.99):
    if actual.shape != expected.shape:
        return {"shape_match": False, "passed": False}
    if not actual.is_floating_point() or not expected.is_floating_point():
        exact = torch.equal(actual, expected)
        return {"shape_match": True, "exact_match": exact, "passed": exact}
    a, e = actual.detach().cpu().double().reshape(-1), expected.detach().cpu().double().reshape(-1)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    result = {"shape_match": True, "finite": finite, "passed": False,
              "max_relative_l2": max_relative_l2, "min_pcc": min_pcc}
    if not finite:
        return result
    relative = ((a-e).norm() / e.norm().clamp_min(1e-30)).item()
    ac, ec = a-a.mean(), e-e.mean()
    denominator = ac.norm()*ec.norm()
    pcc = ((ac*ec).sum()/denominator).item() if denominator > 0 else None
    result.update(relative_l2=relative, pcc=pcc,
                  max_abs=(a-e).abs().max().item() if a.numel() else 0.,
                  passed=relative <= max_relative_l2 and (pcc is None or pcc >= min_pcc))
    return result


def enforce_check(report, name, actual, expected, **thresholds):
    result = compare_tensors(actual, expected, **thresholds)
    report.setdefault("numerical_checks", {})[name] = result
    if not result["passed"]:
        raise ValueError(f"Numerical smoke gate failed: {name}: {result}")
    return result
