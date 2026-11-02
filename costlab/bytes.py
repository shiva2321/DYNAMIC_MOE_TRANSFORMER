"""Model parameter and buffer memory footprint measurement."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import torch
import torch.nn as nn


class ActiveBytes(NamedTuple):
    """Result of active parameter and buffer bytes measurement.

    Attributes:
        total: Sum of unique active parameter and buffer bytes across executed modules.
        by_module: Bytes owned by each executed module that owns any ("(root)" is the model itself).
    """

    total: int
    by_module: dict[str, int]


def total_param_bytes(model: nn.Module) -> int:
    """Calculate total bytes occupied by parameters and buffers of a model.

    Each unique underlying tensor storage (by data_ptr) is counted exactly once.

    Args:
        model: PyTorch module.

    Returns:
        int: Total bytes of parameters and buffers.
    """
    seen: set[int] = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        ptr = tensor.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += tensor.numel() * tensor.element_size()
    return total


def active_param_bytes(
    model: nn.Module,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> ActiveBytes:
    """Measure the parameter and buffer bytes of the modules executed during fn.

    Registers a forward hook on every module of ``model``, leaf or not, runs
    fn(*args, **kwargs), and tallies the parameters and buffers *owned* by each module
    that executed at least once (``recurse=False``, so nothing is counted twice and a
    module that owns parameters and also has children is counted correctly). Each
    tensor storage is counted once (by data_ptr) in the total.

    Limit: the whole tensor of an executed module counts, so a module that keeps all
    its experts in one parameter tensor and indexes one of them reads as fully active.

    Args:
        model: The root PyTorch module.
        fn: Callable running the forward computation.
        *args: Positional arguments for fn.
        **kwargs: Keyword arguments for fn.

    Returns:
        ActiveBytes: namedtuple holding total active bytes and a per-module breakdown.
    """
    called: set[nn.Module] = set()
    hooks = []
    names: dict[nn.Module, str] = {}

    for name, module in model.named_modules():
        names[module] = name or "(root)"
        hooks.append(module.register_forward_hook(lambda m, _inp, _out: called.add(m)))

    try:
        fn(*args, **kwargs)
    finally:
        for h in hooks:
            h.remove()

    seen: set[int] = set()
    total = 0
    by_module: dict[str, int] = {}

    for module in called:
        mod_bytes = 0
        for tensor in list(module.parameters(recurse=False)) + list(
            module.buffers(recurse=False)
        ):
            b = tensor.numel() * tensor.element_size()
            mod_bytes += b
            ptr = tensor.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += b
        if mod_bytes:
            by_module[names[module]] = mod_bytes

    return ActiveBytes(total=total, by_module=by_module)
