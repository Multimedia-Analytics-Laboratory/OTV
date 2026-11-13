"""Outcome-derived supervision on response-prefix boundaries."""

import torch


TARGET_MODES = ("linear", "constant", "sigmoid", "exponential", "wave", "step", "sine")


def make_targets(lengths, response_starts, correct, positions, mode="linear"):
    """Return targets and a mask; u=0 before the response and u=1 at its end.

    This boundary convention is explicit and differs from the legacy trainer's
    first-response-token offset. Sine is an optional new rule, not a paper result.
    """
    if mode not in TARGET_MODES:
        raise ValueError(f"Unsupported target rule: {mode}.")
    if ((response_starts < 0) | (response_starts >= lengths)).any():
        raise ValueError("Every example must have a nonempty response.")
    if not ((correct == 0) | (correct == 1)).all():
        raise ValueError("Correctness labels must be 0 or 1.")
    if positions.ndim == 1:
        positions = positions[None].expand(lengths.shape[0], -1)
    valid = (positions >= response_starts[:, None]) & (positions <= lengths[:, None])
    progress = ((positions - response_starts[:, None]).float()
                / (lengths - response_starts)[:, None]).clamp(0, 1)
    if mode == "constant":
        shape = torch.ones_like(progress)
    elif mode == "sigmoid":
        shape = torch.sigmoid(4 * (progress - .5))
    elif mode == "exponential":
        shape = 2 ** progress - 1
    elif mode == "wave":
        shape = (progress + .1 * torch.sin(6 * torch.pi * progress)).clamp(0, 1)
    elif mode == "step":
        shape = (torch.floor(20 * progress) / 19).clamp(0, 1)
    elif mode == "sine":
        shape = torch.sin(torch.pi * progress / 2)
    else:
        shape = progress
    return .5 + (correct.float()[:, None] - .5) * shape, valid
