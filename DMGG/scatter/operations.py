from typing import cast

import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Broadcast `src` to match `other` along `dim`.

    This is mainly intended for broadcasting 1D index tensors to match the
    shape of a higher-dimensional source tensor.
    No memory allocated beyond the broadcasted view.

    Args:
        src: [N, ] or broadcastable to other.shape after inserting singleton dimensions.
        other: [d_0, ..., d_{k-1}], Reference tensor to match the shape of.
        dim: Dimension of other along which the length of src is placed.

    Return:
        A view of src with other.shape
    """
    if dim < 0:
        dim = other.dim() + dim

    # shape: [1]*dim + [src_len] + [1,...] to match other.dim()
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def _compute_out_size(
    src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int | None
) -> list[int]:
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return size


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = _compute_out_size(src, index, dim, dim_size)
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    else:
        out.zero_()
    return out.scatter_add_(dim, index, src)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
) -> torch.Tensor:
    if src.is_floating_point():
        dtype = src.dtype
    elif torch.is_autocast_enabled():
        dtype = (
            torch.get_autocast_gpu_dtype()
            if src.is_cuda
            else torch.get_autocast_cpu_dtype()
        )
    else:
        dtype = torch.get_default_dtype()
    src = src.to(dtype)

    out_sum = scatter_sum(src, index, dim, dim_size=dim_size)

    count = scatter_sum(
        torch.ones_like(src, dtype=dtype), index, dim, dim_size=dim_size
    )

    # Avoid division by zero
    return out_sum.div(count.clamp(min=1))


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    if src.is_floating_point():
        sentinel = float("-inf")
    else:
        sentinel = torch.iinfo(src.dtype).min

    index = broadcast(index, src, dim)
    if out is None:
        size = _compute_out_size(src, index, dim, dim_size)
        out = torch.full(size, sentinel, dtype=src.dtype, device=src.device)
    else:
        out.fill_(sentinel)

    out.scatter_reduce_(dim, index, src, reduce="amax", include_self=True)
    return out


def scatter_min(
    src: torch.Tensor | torch.Tensor,
    index: torch.Tensor | torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor | torch.Tensor:
    if src.is_floating_point():
        sentinel = float("inf")
    else:
        sentinel = torch.iinfo(src.dtype).max

    index = broadcast(index, src, dim)
    if out is None:
        size = _compute_out_size(src, index, dim, dim_size)
        out = torch.full(size, sentinel, dtype=src.dtype, device=src.device)
    else:
        out.fill_(sentinel)

    out.scatter_reduce_(dim, index, src, reduce="amin", include_self=True)
    return out


def scatter_argmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
) -> torch.LongTensor:
    index = broadcast(index, src, dim)

    # Max value per group
    max_val = scatter_max(src, index, dim, dim_size=dim_size)
    max_val_expanded = max_val.gather(dim, index)
    mask = src == max_val_expanded

    # 원본 dim 위치의 인덱스 범위
    dim_pos = dim if dim >= 0 else dim + src.dim()
    dim_range = torch.arange(src.size(dim_pos), device=src.device, dtype=torch.int64)
    view_shape = [1] * src.dim()
    view_shape[dim_pos] = -1
    dim_range = dim_range.view(view_shape).expand_as(index)

    sentinel = torch.iinfo(torch.int64).max
    masked = torch.where(mask, dim_range, sentinel)

    out = torch.full(max_val.shape, sentinel, device=src.device, dtype=torch.int64)
    out.scatter_reduce_(dim, index, masked, reduce="amin", include_self=True)
    return cast(torch.LongTensor, out)


def scatter_argmin(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
) -> torch.LongTensor:
    index = broadcast(index, src, dim)

    # Min value per group
    min_val = scatter_min(src, index, dim, dim_size=dim_size)
    min_val_expanded = min_val.gather(dim, index)
    mask = src == min_val_expanded

    dim_pos = dim if dim >= 0 else dim + src.dim()
    dim_range = torch.arange(src.size(dim_pos), device=src.device, dtype=torch.int64)
    view_shape = [1] * src.dim()
    view_shape[dim_pos] = -1
    dim_range = dim_range.view(view_shape).expand_as(index)

    sentinel = torch.iinfo(torch.int64).max
    masked = torch.where(mask, dim_range, sentinel)

    out = torch.full(min_val.shape, sentinel, device=src.device, dtype=torch.int64)
    out.scatter_reduce_(dim, index, masked, reduce="amin", include_self=True)
    return cast(torch.LongTensor, out)


def scatter_logsumexp(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)

    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    size = list(src.size())
    size[dim if dim >= 0 else dim + src.dim()] = dim_size

    max_per_index = scatter_max(src, index, dim, dim_size=dim_size)
    max_per_src = max_per_index.gather(dim, index)
    recentered = src - max_per_src
    recentered = recentered.masked_fill(torch.isnan(recentered), float("-inf"))

    sum_per_index = scatter_sum(recentered.exp(), index, dim, dim_size=dim_size)
    return sum_per_index.log().add(max_per_index)


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)

    max_per_index = scatter_max(src, index, dim, dim_size=dim_size)
    max_per_src = max_per_index.gather(dim, index)

    recentered = src - max_per_src
    recentered = recentered.masked_fill(torch.isnan(recentered), float("-inf"))

    exp_vals = recentered.exp()

    sum_per_index = scatter_sum(exp_vals, index, dim, dim_size=dim_size)
    sum_per_src = sum_per_index.gather(dim, index)

    return exp_vals / sum_per_src
