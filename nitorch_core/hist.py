import torch
from nitorch_core import py, dtypes
from nitorch_core.tensors import movedim, backend, to_max_backend
from nitorch_core.constants import eps
from interpol import grid_push, grid_pull, grid_grad, grid_count


def histc(x, n=64, min=None, max=None, dim=None, keepdim=False, weights=None,
          order=1, bound='replicate', extrapolate=False, dtype=None):
    """Batched + differentiable histogram computation

    Parameters
    ----------
    x : tensor_like
        Input tensor.
    n : int, default=64
        Number of bins.
    min : float or tensor_like, optional
        Left edge of the histogram.
        Must be broadcastable to the input batch shape.
    max : float or tensor_like, optional
        Right edge of the histogram.
        Must be broadcastable to the input batch shape.
    dim : [sequence of] int, default=all
        Dimensions along which to compute the histogram
    keepdim : bool, default=False
        Keep singleton dimensions.
    weights : tensor, optional
        Observation weights
    order : {0..7}, default=1
        B-spline order encoding the histogram
    bound : bound_like, default='replicate'
        Boundary condition (only used when order > 1 or extrapolate is True)
    extrapolate : bool, default=False
        If False, discard data points that fall outside of [min, max]
        If True, use `bound` to assign them to a bin.
    dtype : torch.dtype, optional
        Output data type.
        Default: same as x unless it is not a floating point type, then
        `torch.get_default_dtype()`

    Returns
    -------
    h : (..., n) tensor
        Count histogram

    """
    # reshape as [batch, pool]]
    x = torch.as_tensor(x)
    if weights is not None:
        dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
        weights = torch.as_tensor(weights, dtype=dtype, device=x.device).expand(x.shape)
    if dim is None:
        x = x.reshape([1, -1])
        batch = []
        if weights is not None:
            weights = weights.reshape([1, -1])
    else:
        dim = py.make_list(dim)
        odim = list(range(-len(dim), 0))
        inshape = x.shape
        x = movedim(x, dim, odim)
        batch = x.shape[:-len(dim)]
        pool = x.shape[-len(dim):]
        x = x.reshape([-1, py.prod(pool)])
        if weights is not None:
            weights = weights.reshape([-1, py.prod(pool)])

    # compute limits
    if min is None:
        min = x.min(dim=-1, keepdim=True).values
    else:
        min = torch.as_tensor(min)
        min = min.expand(batch).reshape([-1, 1])
    if max is None:
        max = x.max(dim=-1, keepdim=True).values
    else:
        max = torch.as_tensor(max)
        max = max.expand(batch).reshape([-1, 1])

    # convert intensities to coordinates
    # (min -> -0.5  // max -> n-0.5)
    if not dtypes.dtype(x.dtype).is_floating_point:
        ftype = torch.get_default_dtype()
        x = x.to(ftype)
    x = x.clone()
    x = x.mul_(n / (max - min)).add_(n / (1 - max / min)).sub_(0.5)

    # push data into the histogram
    if not extrapolate:
        # hidden feature: tell pullpush to use +/- 0.5 tolerance when
        # deciding if a coordinate is inbounds.
        extrapolate = 2
    if weights is None:
        # count == push an image of ones
        h = grid_count(x[:, :, None], [n], order, bound, extrapolate)[:, 0, ]
    else:
        # push weights
        h = grid_push(weights[:, None, :], x[:, :, None], [n], order, bound, extrapolate)[:, 0, ]

    # reshape
    h = h.to(dtype)
    if keepdim:
        oshape = list(inshape)
        for d in dim:
            oshape[d] = 1
        oshape += [n]
    else:
        oshape = [*batch, n]
    h = h.reshape(oshape)
    return h


def histc2(x, n=64, min=None, max=None, dim=None, keepdim=False,
           order=1, bound='replicate', extrapolate=False, dtype=None):
    """Batched + differentiable joint histogram computation

    Parameters
    ----------
    x : (..., 2) tensor_like
        Input tensor.
    n : int or (int, int), default=64
        Number of bins.
    min : float or tensor_like, optional
        Left edge of the histogram.
        Must be broadcastable to (*batch, 2).
    max : float or tensor_like, optional
        Right edge of the histogram.
        Must be broadcastable to (*batch, 2).
    dim : [sequence of] int, default=all
        Dimensions along which to compute the histogram
    keepdim : bool, default=False
        Keep singleton dimensions.
    order : {0..7}, default=1
        B-spline order encoding the histogram
    bound : bound_like, default='replicate'
        Boundary condition (only used when order > 1 or extrapolate is True)
    extrapolate : bool, default=False
        If False, discard data points that fall outside of [min, max]
        If True, use `bound` to assign them to a bin.
    dtype : torch.dtype, optional
        Output data type.
        Default: same as x unless it is not a floating point type, then
        `torch.get_default_dtype()`

    Returns
    -------
    h : (..., n) tensor
        Count histogram

    """
    n = py.make_list(n, 2)

    # reshape as [batch, pool, 2]]
    x = torch.as_tensor(x)
    bck = backend(x)
    if dim is None:
        x = x.reshape([1, -1, 2])
        batch = []
    else:
        dim = py.make_list(dim)
        if -1 in dim or (x.dim()-1) in dim:
            raise ValueError('Cannot pool along last dimension')
        odim = list(range(-len(dim)-1, -1))
        inshape = x.shape
        x = movedim(x, dim, odim)
        batch = x.shape[:-len(dim)-1]
        pool = x.shape[-len(dim)-1:-1]
        x = x.reshape([-1, py.prod(pool), 2])

    # compute limits
    if min is None:
        min = x.detach().min(dim=-2, keepdim=True).values
    else:
        min = torch.as_tensor(min, **bck)
        min = min.expand([*batch, 2]).reshape([-1, 1, 2])
    if max is None:
        max = x.detach().max(dim=-2, keepdim=True).values
    else:
        max = torch.as_tensor(max, **bck)
        max = max.expand([*batch, 2]).reshape([-1, 1, 2])

    # convert intensities to coordinates
    # (min -> -0.5  // max -> n-0.5)
    if not dtypes.dtype(x.dtype).is_floating_point:
        ftype = torch.get_default_dtype()
        x = x.to(ftype)
    x = x.clone()
    nn = torch.as_tensor(n, dtype=x.dtype, device=x.device)
    x = x.mul_(nn / (max - min)).add_(nn / (1 - max / min)).sub_(0.5)

    # push data into the histogram
    if not extrapolate:
        # hidden feature: tell pullpush to use +/- 0.5 tolerance when
        # deciding if a coordinate is inbounds.
        extrapolate = 2
    h = grid_count(x[:, None], n, order, bound, extrapolate)[:, 0]

    # reshape
    h = h.to(dtype)
    if keepdim:
        oshape = list(inshape)
        for d in dim:
            oshape[d] = 1
        oshape += n
    else:
        oshape = [*batch, *n]
    h = h.reshape(oshape)
    return h


def _hist_to_quantile(hist, q):
    """Compute quantiles from a cumulative histogram.

    Parameters
    ----------
    hist : (B, K) tensor
        Strictly monotonic cumulative histogram.
        B = batch size, K = number of bins
    q : (Q,) tensor
        Quantiles to compute.
        Q = number of quantiles.

    Returns
    -------
    values : (B, Q) tensor
        Quantile values, expressed in bins.
        They can be converted to values by `vmin + values * bin_width`.

    """
    hist, q = to_max_backend(hist, q, force_float=True)
    # compute the distance between discrete quantiles and target quantile
    hist = hist[:, None, :] - q[None, :, None]
    # find discrete quantile nearest to target quantile
    tmp = hist.clone()
    tmp[tmp < 0] = float('inf')  # approach from below
    delta1, binq = tmp.min(dim=-1)
    # compute left weight (this is super ugly)
    delta0 = hist.neg().gather(-1, (binq - 1).clamp_min_(0)[..., None])[..., 0]
    delta0[binq == 0] = q.expand(delta0.shape)[binq == 0]
    del hist
    # compute interpolation weights
    delta0, delta1 = (delta1 / (delta0 + delta1), delta0 / (delta0 + delta1))
    # interpolate value
    q = delta0 * binq + delta1 * (binq + 1)
    return q


def quantile(input, q, dim=None, keepdim=False, bins=None, mask=None, *, out=None):
    """Compute quantiles.

    Parameters
    ----------
    input : tensor_like
        Input Tensor.
    q : float or (K,) tensor_like
        Values in [0, 1]: quantiles to computes
    dim : [sequence of] int, default=all
        Dimensions to reduce
    keepdim : bool, default=False
        Whether to squeeze reduced dimensions.
    bins : int, optional
        Number of histogram bins to use for fast quantile computation.
        By default: exact (but slow) computation using sorting.
    out : tensor, optional
        Output placeholder.

    Returns
    -------
    quant : (..., [K]) tensor
        Quantiles

    """
    def torch_is_recent():
        version = torch.__version__.split('.')
        version = (int(version[0]), int(version[1]))
        return version[0] > 2 or (version[0] == 1 and version[1] >= 7)

    input, q = to_max_backend(input, q)
    dim = py.make_list([] if dim is None else dim)
    # if torch_is_recent() and len(dim) < 2 and not bins:
    #     dim = dim[0] if dim else None
    #     return torch.quantile(input, q, dim=dim, keepdim=keepdim, out=out)

    # ------------------
    # our implementation
    # ------------------

    # reshape as (batch, pool)
    inshape = input.shape
    if mask is not None:
        mask = mask.expand(inshape)
    if not dim:
        if mask is not None:
            mask = mask.reshape([1, -1])
        input = input.reshape([1, -1])
        batch = []
    else:
        odim = list(range(-len(dim), 0))
        input = movedim(input, dim, odim)
        batch = input.shape[:-len(dim)]
        pool = input.shape[-len(dim):]
        input = input.reshape([-1, py.prod(pool)])
        if mask is not None:
            mask = movedim(mask, dim, odim).reshape([-1, py.prod(pool)])

    q_scalar = q.dim() == 0
    q = q.reshape([-1]).clone()
    if not bins and mask is None:
        # sort and sample
        input, _ = input.sort(-1)
        q = q.mul_(input.shape[-1]-1)
        q = grid_pull(input[None], q[None, :, None], 1, 'replicate', 0)[0]
    elif not bins:
        input, index = input.sort(-1)
        mask = mask.gather(-1, index)
        mask = mask.cumsum(-1) / mask.sum(-1, keepdim=True)
        mask[:, -1] = 1
        q = _hist_to_quantile(mask, q)
        q = grid_pull(input[:, None], q[:, :, None], 1, 'replicate', 0)[:, 0]
    else:
        # compute cumulative histogram
        min = input.min(-1).values
        max = input.max(-1).values
        bin_width = (max-min)/bins
        hist = histc(input, bins, dim=-1, min=min, max=max, weights=mask)
        del max, input
        hist += eps(hist.dtype)  # ensures monotonicity
        hist = hist.cumsum(-1) / hist.sum(-1, keepdim=True)
        hist[..., -1] = 1  # avoid rounding errors
        # interpolate quantile value
        q = _hist_to_quantile(hist, q)
        q = min[:, None] + q * bin_width[:, None]

    # reshape
    if keepdim:
        if not dim:
            oshape = [1] * len(inshape)
        else:
            oshape = list(inshape)
            for d in dim:
                oshape[d] = 1
        oshape += [q.shape[-1]]
    else:
        oshape = [*batch, q.shape[-1]]
    q = q.reshape(oshape)
    if q_scalar:
        q = q.squeeze(-1)

    if out:
        out.reshape(q.shape).copy_(q)
    return q
