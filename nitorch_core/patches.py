__all__ = ['unfold', 'fold', 'apply_patchwise']
import torch
import math as pymath
import itertools
from nitorch_core import dtypes
from nitorch_core.py import make_list
from nitorch_core.padding import pad, ensure_shape
from nitorch_core.extra import movedim


def unfold(inp, kernel_size, stride=None, collapse=False):
    """Extract patches from a tensor.

    Parameters
    ----------
    inp : (..., *spatial) tensor
        Input tensor.
    kernel_size : [sequence of] int
        Patch shape.
    stride : [sequence of] int, default=`kernel_size`
        Stride.
    collapse : bool or 'view', default=False
        Collapse the original spatial dimensions.
        If 'view', forces collapsing to use the view mechanism, which ensures
        that no data copy is triggered. This can fail if the tensor's
        strides do not allow these dimensions to be collapsed.

    Returns
    -------
    out : (..., *spatial_out, *kernel_size) tensor
        Output tensor of patches.
        If `collapse`, the output spatial dimensions (`spatial_out`)
        are flattened.

    """
    inp = torch.as_tensor(inp)
    kernel_size = make_list(kernel_size)
    dim = len(kernel_size)
    batch_dim = inp.dim() - dim
    stride = make_list(stride, dim)
    stride = [st or sz for st, sz in zip(stride, kernel_size)]
    for d, (sz, st) in enumerate(zip(kernel_size, stride)):
        inp = inp.unfold(dimension=batch_dim+d, size=sz, step=st)
    if collapse:
        batch_shape = inp.shape[:-dim*2]
        if collapse == 'view':
            inp = inp.view([*batch_shape, -1, *kernel_size])
        else:
            inp = inp.reshape([*batch_shape, -1, *kernel_size])
    return inp


def fold(inp, dim=None, stride=None, shape=None, collapsed=False,
         reduction='mean'):
    """Reconstruct a tensor from patches.

    .. warning: This function only works if `kernel_size <= 2*stride`.

    Parameters
    ----------
    inp : (..., *spatial, *kernel_size) tensor
        Input tensor of patches
    dim : int
        Length of `kernel_size`.
    stride : [sequence of] int, default=`kernel_size`
        Stride.
    shape : sequence of int, optional
        Output shape. By default, it is computed from `spatial`,
        `stride` and `kernel_size`. If the output shape is larger than
        the computed shape, zero-padding is used.
        This parameter is mandatory if `collapsed = True`.
    collapsed : 'view' or bool, default=False
        Whether the spatial dimensions are collapsed in the input tensor.
        If 'view', use `view` instead of `reshape`, which will raise an
        error instead of triggering a copy when dimensions cannot be
        collapsed in a contiguous way.
    reduction : {'mean', 'sum', 'min', 'max'}, default='mean'
        Method to use to merge overlapping patches.

    Returns
    -------
    out : (..., *shape) tensor
        Folded tensor

    """
    def recon(x, stride):
        dim = len(stride)
        inshape = x.shape[-2*dim:-dim]
        batch_shape = x.shape[:-2*dim]
        indim = list(reversed(range(-1, -2 * dim - 1, -1)))
        outdim = (list(reversed(range(-2, -2 * dim - 1, -2))) +
                  list(reversed(range(-1, -2 * dim - 1, -2))))
        x = movedim(x, indim, outdim)
        outshape = [i * k for i, k in zip(inshape, stride)]
        x = x.reshape([*batch_shape, *outshape])
        return x

    inp = torch.as_tensor(inp)
    if torch.is_tensor(shape):
        shape = shape.tolist()
    dim = dim or (len(shape) if shape else None)
    if not dim:
        raise ValueError('Cannot guess dim from inputs')
    kernel_size = inp.shape[-dim:]
    stride = make_list(stride, len(kernel_size))
    stride = [st or sz for st, sz in zip(stride, kernel_size)]
    if any(sz > 2*st for st, sz in zip(stride, kernel_size)):
        # I only support overlapping of two patches (along a given dim).
        # If the kernel  is too large, more than two patches can overlap
        # and this function fails.
        raise ValueError('This function only works if kernel_size <= 2*stride')
    if not shape:
        if collapsed:
            raise ValueError('`shape` is mandatory when `collapsed=True`')
        inshape = inp.shape[-dim*2:-dim]
        shape = [(i-1)*st + sz
                 for i, st, sz in zip(inshape, stride, kernel_size)]
    else:
        inshape = [(o - sz) // st + 1
                   for o, st, sz in zip(shape, stride, kernel_size)]

    if collapsed:
        batch_shape = inp.shape[:-dim-1]
        inp = inp.reshape([*batch_shape, *inshape, *kernel_size])
    batch_shape = inp.shape[:-2*dim]

    # When the stride is equal to the kernel size, folding is easy
    # (it is obtained by shuffling dimensions and reshaping)
    # However, in the more general case, patches can overlap or,
    # conversely, have gaps between them. In the first case,
    # overlapping values must be reduced somehow. In the second case,
    # patches must be padded.

    # 1) padding (stride > kernel_size)
    padding = [max(0, st - sz) for st, sz in zip(stride, kernel_size)]
    padding = [0] * (inp.dim() - dim) + padding
    inp = pad(inp, padding, side='right')
    stride = [(st if st < sz else sz) for st, sz in zip(stride, kernel_size)]
    kernel_size = inp.shape[-dim:]

    # 2) merge overlaps
    overlap = [max(0, sz - st) for st, sz in zip(stride, kernel_size)]
    if any(o != 0 for o in overlap):
        slicer = [slice(None)] * (inp.dim() - dim)
        slicer += [slice(k) for k in stride]
        out = inp[tuple(slicer)].clone()
        if reduction == 'mean':
            count = inp.new_ones([*inshape, *stride], dtype=torch.int)
            fn = 'sum'
        else:
            count = None
            fn = reduction

        # ! a bit of padding to save the last values
        padding = [1 if o else 0 for o in overlap] + [0] * dim
        if count is not None:
            count = pad(count, padding, side='right')
        padding = [0] * (out.dim() - 2*dim) + padding
        value = (dtypes.dtype(inp.dtype).min if fn == 'max' else
                 dtypes.dtype(inp.dtype).max if fn == 'min' else 0)
        out = pad(out, padding, value=value, side='right')

        slicer1 = [slice(-1 if o else None) for o in overlap]
        slicer2 = [slice(None)] * dim
        slicer1 += [slice(st) for st in stride]
        slicer2 += [slice(st) for st in stride]

        import itertools
        overlaps = itertools.product(*[[0, 1] if o else [0] for o in overlap])
        for overlap in overlaps:
            front_slicer = list(slicer1)
            back_slicer = list(slicer2)
            for d, o in enumerate(overlap):
                if o == 0:
                    continue
                front_slicer[-dim+d] = slice(o)
                front_slicer[-2*dim+d] = slice(1, None)
                back_slicer[-dim+d] = slice(-o, None)
                back_slicer[-2*dim+d] = slice(None)
            if count is not None:
                count[tuple(front_slicer)] += 1
            front_slicer = (Ellipsis, *front_slicer)
            back_slicer = (Ellipsis, *back_slicer)

            if fn == 'sum':
                out[front_slicer] += inp[back_slicer]
            elif fn == 'max':
                out[front_slicer] = torch.max(out[front_slicer], inp[back_slicer])
            elif fn == 'min':
                out[front_slicer] = torch.min(out[front_slicer], inp[back_slicer])
            else:
                raise ValueError(f'Unknown reduction {reduction}')
        if count is not None:
            out /= count
    else:
        out = inp.clone()

    # end) reshape
    out = recon(out, stride)
    out = ensure_shape(out, [*batch_shape, *shape], side='right')

    return out


def apply_patchwise(fn, img, patch=256, overlap=None, batchout=None,
                    dim=None, device=None, bound='dct2'):
    """Apply a function to a tensor patch-wise

    Parameters
    ----------
    fn : callable[Tensor] -> Tensor
        Function to apply, should return a tensor with the same shape
        as the input tensor.
    img : (*batchin, *spatial) tensor
        Input tensor
    patch : [list of] int
        Patch size per dimension
    overlap : [list of] int, optional
        Amount of overlap across patches.
        By default, half of the patch overlaps.
    batchout : list[int], default=`batchin`
        Output batch-size.
    dim : int, optional
        Number of spatial dimensions. By default, try to guess.
    device : torch.device, default=`img.device`
        Run patches on this device
    bound : str, default='dct2'
        Boundary conditions used to pad the input tensor, if needed.

    Returns
    -------
    out : Output tensor

    """
    dim = dim or (len(patch) if hasattr(patch, '__len__') else img.dim())
    patch = make_list(patch, dim)
    overlap = make_list(overlap, dim)
    overlap = [p//2 if o is None else o for o, p in zip(overlap, patch)]
    opatch = [p-o for o, p in zip(overlap, patch)]

    batchin, spatial = img.shape[:-dim], img.shape[-dim:]
    if batchout is None:
        batchout = batchin
    out = img.new_zeros([*batchout, *spatial])

    padsize = [_ for o in overlap for _ in (o//2, o)]
    img = pad(img, padsize, mode=bound)

    if device and torch.device(device).type == 'cuda' and img.device.type == 'cpu':
        img = img.pin_memory()
        out = out.pin_memory()

    nb_patch = [int(pymath.ceil(s/p)) for s, p in zip(spatial, opatch)]
    indices = itertools.product(*[range(n) for n in nb_patch])
    for index in indices:
        # extract input patch (with overlap)
        img_slicer = [slice(op*i, op*i+p)
                      for op, p, i in zip(opatch, patch, index)]
        img_slicer = (Ellipsis, *img_slicer)
        block = img[img_slicer].to(device)
        # apply function
        block = fn(block).detach()
        # extract center of output patch and assign to output volume
        out_shape = [min(s, op*(i+1)) - op*i
                     for s, op, i in zip(spatial, opatch, index)]
        blk_slicer = [slice(o//2, o//2+s) for o, s in zip(overlap, out_shape)]
        blk_slicer = (Ellipsis, *blk_slicer)
        out_slicer = [slice(op*i, op*(i+1)) for op, i in zip(opatch, index)]
        out_slicer = (Ellipsis, *out_slicer)
        out[out_slicer] = block[blk_slicer].to(out)
    return out
