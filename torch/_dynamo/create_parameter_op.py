# mypy: allow-untyped-defs
import threading
from contextlib import contextmanager

import torch

doc = """
This is used when dynamo traces torch.nn.Parameter, which normally would not trace properly
with AOTAutograd.  We instead create a placeholder torch.nn.Parameter before the graph, which
becomes a graph arg and has no storage backing it.  At the point in the graph where the parameter
actually should be created we mutate this sacrificial placeholder into it.  This allows gradients
to flow into the parameter as if it were an input to the graph (which is the only thing we are
allowed to compute gradients on).
""".strip()


class TracableCreateParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, placeholder):
        assert not tensor.requires_grad
        if isinstance(tensor, torch.distributed._tensor.api.DTensor):
            with torch.no_grad():
                # DTensor doesn't have .set_(), so have to use .copy_()
                placeholder.copy_(tensor)
        else:
            placeholder.set_(tensor)
        return placeholder

    @staticmethod
    def backward(ctx, grad):
        return None, grad  # grad flows to placeholder


def tracable_create_parameter(tensor, placeholder):
    with torch.set_grad_enabled(placeholder.requires_grad):
        out = TracableCreateParameter.apply(tensor, placeholder)
    return out


def new_parameter_placeholder(size, dtype, device, requires_grad):
    """Create a placeholder to be passed to the above functions"""
    result = torch.nn.Parameter(
        torch.empty(size, dtype=dtype, device=device), requires_grad=requires_grad
    )
    # TODO(jansel): alloc followed by free is inefficient, need a way to allocate an unbacked tensor.
    # Allocating a zero tensor would causes assert failures in autograd.
    result.untyped_storage().resize_(0)
    return result


def new_parameter_placeholder_dtensor(
    local_tensor_size,
    local_tensor_dtype,
    local_tensor_device,
    requires_grad,
    device_mesh,
    placements,
):
    """Create a placeholder to be passed to the above functions"""
    data_tensor = torch.empty(
        local_tensor_size, dtype=local_tensor_dtype, device=local_tensor_device
    )
    # data_tensor.untyped_storage().resize_(0)  # this causes segfault, need to figure out why
    # NOTE: allocate a placeholder nn.Parameter(DTensor), whose content will get swapped out in TracableCreateParameter.forward
    data_tensor = torch.distributed._tensor.api.DTensor.from_local(
        data_tensor,
        device_mesh=device_mesh,
        placements=placements,
    )
    result = torch.nn.Parameter(data_tensor, requires_grad=requires_grad)
    return result


_TLS = threading.local()


@contextmanager
def do_not_convert_to_tracable_parameter():
    old_flag = getattr(_TLS, "convert_tracable_parameter", True)
    _TLS.convert_tracable_parameter = False
    try:
        yield False
    finally:
        _TLS.convert_tracable_parameter = old_flag


def can_convert_to_tracable_parameter():
    return getattr(_TLS, "convert_tracable_parameter", True)
