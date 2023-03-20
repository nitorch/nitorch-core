"""PyTorch utilities."""

import torch
from torch import Tensor
import math as pymath
import itertools
import numbers
import os
import random
from typing import Optional, List
import contextlib


from nitorch_core.constants import inf
from nitorch_core import py, dtypes, bounds, jit
from nitorch_core.optionals import numpy as np
from nitorch_core.version import torch_version


def as_tensor(input, dtype=None, device=None):
    """Convert object to tensor.

    This function expands ``torch.as_tensor`` by accepting nested lists
    of tensors. It works by recursively stacking elements of the input
    list. It is probably much slower than ``torch.as_tensor``.

    Parameters
    ----------
    input : tensor_like
        Input object: tensor or (nested) list/tuple of tensors/scalars
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output tensor.

    """
    # TODO: if torch >= 1.6, use` torch.as_tensor`
    #   I have to clean uses of `utils.as_tensor` first because the
    #   order of arguments is a bit different (I think it is device then
    #   dtype in torch)
    def _stack(x, dtype, device):
        if torch.is_tensor(x):
            return x.to(device if device is not None else x.device,
                        dtype if dtype is not None else x.dtype)
        else:
            if isinstance(x, (list, tuple)):
                subs = [_stack(e, dtype, device) for e in x]
                backend = max_backend(*subs)
                subs = [elem.to(**backend) for elem in subs]
                return torch.stack(subs)
            else:
                return torch.as_tensor(x, dtype=dtype, device=device)

    return _stack(input, dtype, device)


def make_vector(input, n=None, crop=True, *args, 
                dtype=None, device=None, **kwargs):
    """Ensure that the input is a (tensor) vector and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output vector.

    """
    input = as_tensor(input, dtype=dtype, device=device).flatten()
    if n is None:
        return input
    if n is not None and input.numel() >= n:
        return input[:n] if crop else input
    has_default = False
    if args:
        has_default = True
        default = args[0]
    elif 'default' in kwargs:
        has_default = True
        default = kwargs['default']
    if has_default:
        return ensure_shape(input, n, mode='constant', value=default)
    else:
        return ensure_shape(input, n, mode='replicate')
        

def unsqueeze(input, dim=0, ndim=1):
    """Adds singleton dimensions to a tensor.

    This function expands `torch.unsqueeze` with additional options.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int, default=0
        Position at which to insert singleton dimensions.
    ndim : int, default=1
        Number of singleton dimensions to insert.

    Returns
    -------
    output : tensor
        Tensor with additional singleton dimensions.
    """
    for _ in range(ndim):
        input = torch.unsqueeze(input, dim)
    return input


def squeeze(input, dim=0, ndim=1):
    """Removes singleton dimensions to a tensor.

    This function expands `torch.squeeze` with additional options.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int, default=0
        Position at which to drop singleton dimensions.
    ndim : int, default=1
        Number of singleton dimensions to drop.

    Returns
    -------
    output : tensor
        Tensor with singleton dimensions removed.
    """
    for _ in range(ndim):
        input = torch.squeeze(input, dim)
    return input


def invert_permutation(perm):
    """Return the inverse of a permutation

    Parameters
    ----------
    perm : (..., N) tensor_like
        Permutations. A permutation is a shuffled set of indices.

    Returns
    -------
    iperm : (..., N) tensor
        Inverse permutation.

    Examples
    --------
    >>> import torch
    >>> from nitorch.core.utils import invert_permutation
    >>> perm = [0, 2, 3, 1]
    >>> a = torch.rand((len(perm),))
    >>> permuted_a = a[perm]
    >>> recovered_a = permuted_a[invert_permutation(perm)]
    >>> assert((a == recovered_a).all())

    """
    perm = torch.as_tensor(perm)
    shape = perm.shape
    device = perm.device
    perm = perm.reshape([-1, shape[-1]])
    n = perm.shape[-1]
    k = perm.shape[0]
    identity = torch.arange(n, dtype=torch.long, device=device)[None, ...]
    identity = identity.expand(k, n)  # Repeat without allocation
    iperm = torch.empty_like(perm).scatter_(-1, perm, identity)
    iperm = iperm.reshape(shape)
    return iperm


def shiftdim(x, n=None):
    """Shift the dimensions of x by n.

    Parameters
    ----------
        x : torch.Tensor
            Input tensor.
        n : int, default=None
            Shift.
            * When N is positive, `shiftdim` shifts the dimensions to
              the left and wraps the N leading dimensions to the end.
            * When N is negative, `shiftdim` shifts the dimensions to
              the right and pads with singletons.
            * When N is None, `shiftdim` removes all leading singleton
              dimensions. The number of removed dimensions is returned
              as well.

    Returns
    -------
        x : torch.Tensor
            Output tensor.
        n : int, if n is None
            Number of removed dimensions

    """
    if n is None:
        shape = torch.as_tensor(x.size())
        n = (shape != 1).nonzero()
        if n.numel() == 0:
            n = x.dim()
            x = x.reshape([])
        else:
            n = n[0]
            x = x.reshape(shape[n:].tolist())
        return x, n
    elif n < 0:
        x = x.reshape((1,)*(-n) + x.size())
    elif n > 0:
        n = n % x.dim()
        x = x.permute(tuple(range(n, x.dim())) + tuple(range(n)))
    return x


if hasattr(torch, 'movedim'):
    fast_movedim = torch.movedim
else:
    def fast_movedim(input, source, destination):
        """Move the position of exactly one dimension"""
        dim = input.dim()

        source = dim + source if source < 0 else source
        destination = dim + destination if destination < 0 else destination
        permutation = list(range(dim))
        del permutation[source]
        permutation.insert(destination, source)
        return input.permute(*permutation)


def movedim(input, source, destination):
    """Moves the position of one or more dimensions

    Other dimensions that are not explicitly moved remain in their
    original order and appear at the positions not specified in
    destination.

    Parameters
    ----------
    input : tensor
        Input tensor
    source : int or sequence[int]
        Initial positions of the dimensions
    destination : int or sequence[int]
        Output positions of the dimensions.

        If a single destination is provided:
        - if it is negative, the last source dimension is moved to
          `destination` and all other source dimensions are moved to its left.
        - if it is positive, the first source dimension is moved to
          `destination` and all other source dimensions are moved to its right.

    Returns
    -------
    output : tensor
        Tensor with moved dimensions.

    """
    input = torch.as_tensor(input)
    perm = py.move_to_permutation(input.dim(), source, destination)
    return input.permute(*perm)


def movedim_front2back(tensor, dim):
    """Move the first N dimensions to the back"""
    dims = list(range(tensor.dim()))
    perm = dims[dim:] + dims[:dim]
    return tensor.permute(*perm)


def movedim_back2front(tensor, dim):
    """Move the last N dimensions to the front"""
    dims = list(range(tensor.dim()))
    perm = dims[-dim:] + dims[:-dim]
    return tensor.permute(*perm)


def moveelem(input, source, destination, dim=-1):
    """Move elements in a tensor

    Parameters
    ----------
    input : tensor
    source : [sequence of] int
    destination : [sequence of] int
    dim : int, default=-1

    Returns
    -------
    output : tensor

    """
    perm = py.move_to_permutation(input.shape[dim], source, destination)
    perm = torch.as_tensor(perm, dtype=torch.long, device=input.device)
    return input.index_select(dim, perm)


def to(*args, dtype=None, device=None):
    """Move/convert to a common dtype or device.

    Parameters
    ----------
    *args : tensor_like
        Input tensors or tensor-like objects
    dtype : str or torch.dtype, optional
        Target data type
    device : str or torch.device, optional
        Target device

    Returns
    -------
    *args : tensor_like
        Converted tensors

    """
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype, device=device)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype, device=device)
                     if arg is not None else arg for arg in args)


def to_max_backend(*args, force_float=False, dtype=None, device=None):
    """Move to a common dtype and device.

    See `max_dtype` and `max_device`.

    Parameters
    ----------
    *args : tensor_like
    force_float : bool, default=False

    Returns
    -------
    *args_to : tensor

    """
    if len(args) == 0:
        return
    dtype = max_dtype(*args, dtype, force_float=force_float)
    device = max_device(*args, device)
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype, device=device)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype, device=device)
                     if arg is not None else arg for arg in args)


def to_max_device(*args):
    """Move to a common device.

    See `max_device`.

    Parameters
    ----------
    *args : tensor_like

    Returns
    -------
    *args_to : tensor

    """
    if len(args) == 0:
        return
    device = max_device(*args)
    if len(args) == 1:
        return torch.as_tensor(args[0], device=device)
    else:
        return tuple(torch.as_tensor(arg, device=device)
                     for arg in args)


def to_max_dtype(*args):
    """Move to a common data type.

    See `max_dtype`.

    Parameters
    ----------
    *args : tensor_like

    Returns
    -------
    *args_to : tensor

    """
    if len(args) == 0:
        return
    dtype = max_dtype(*args)
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype)
                     for arg in args)


def backend(x):
    """Return the backend (dtype and device) of a tensor

    Parameters
    ----------
    x : tensor

    Returns
    -------
    dict with keys 'dtype' and 'device'

    """
    return dict(dtype=x.dtype, device=x.device)


def max_backend(*args, dtype=None, device=None):
    """Get the (max) dtype and device.

    Parameters
    ----------
    args : tensors

    Returns
    -------
    dict with keys 'dtype' and 'device'

    """
    return dict(dtype=max_dtype(*args, dtype),
                device=max_device(*args, device))


def max_device(*args):
    """Find a common device for all inputs.

    If at least one input object is on a CUDA device:
        * if all cuda object are on the same cuda device, return it
        * if some objects are on different cuda devices, return
          `device('cuda')` without an index.
    Else, return device('cpu') or None.

    Parameters
    ----------
    *args : tensor_like or device_like

    Returns
    -------
    device : torch.device

    """
    from .optionals import numpy as np
    is_array = lambda x: (isinstance(x, np.ndarray) if np else False)
    is_tensor = torch.is_tensor

    def select_device(*many_devices):
        if len(many_devices) == 0:
            return None
        elif len(many_devices) == 1:
            return many_devices[0]
        device1, device2, *many_devices = many_devices
        if len(many_devices) > 0:
            return select_device(select_device(device1, device2), *many_devices)
        if device1 is None:
            return device2
        elif device2 is None:
            return device1
        elif device1.type == 'cuda' and device2.type != 'cuda':
            return device1
        elif device2.type == 'cuda' and device1.type != 'cuda':
            return device2
        elif device1.index is None:
            return device2
        elif device2.index is None:
            return device1
        elif device1.index == device2.index:
            return device1
        else:
            return torch.device('cuda')

    def explore_device(x):
        if x is None:
            return None
        if isinstance(x, (torch.device, str)):
            return torch.device(x)
        elif is_tensor(x):
            return x.device
        elif is_array(x) or isinstance(x, numbers.Number):
            # numpy/builtin type: None
            return None
        else:
            # assume it is a sequence: check what we find in there
            devices = [explore_device(elem) for elem in x]
            return select_device(*devices)

    return explore_device(args)


def max_dtype(*args, force_float=False):
    """Find the maximum data type from a series of inputs.

    The returned dtype is the best one to use for upcasting the objects.

        * Tensors and arrays have priority python objects.
        * Tensors and arrays with non-null dimensionality have priority
          over scalars.
        * If any of the torch/numpy objects have a floating point type
          a floating point type is returned.
        * If any of the objects is complex, a complex type is returned.
        * If all torch/numpy objects have an integer type and there is
          an integer type that avoids overflowing, it is returned.
        * If no integer type that ensures underflowing exists, the default
          floating point data type is returned.
        * If `force_float is True`, a floating point data type is returned
          even if all input objects have an integer data type.

    Parameters
    ----------
    *args : tensor_like or type_like
    force_float : bool, default=False

    Returns
    -------
    dtype : torch.dtype

    """
    from .optionals import numpy as np
    is_array = lambda x: (isinstance(x, np.ndarray) if np else False)
    is_tensor = torch.is_tensor
    is_np_dtype = lambda x: ((isinstance(x, np.dtype) or
                                 (isinstance(x, type) and
                                  issubclass(x, np.number)))
                                if np else False)
    is_torch_dtype = lambda x: isinstance(x, torch.dtype)
    is_py_dtype = lambda x: isinstance(x, type) and issubclass(x, numbers.Number)
    is_dtype = lambda x: is_torch_dtype(x) or is_np_dtype(x) or is_py_dtype(x)

    def upcast(*many_types):
        if len(many_types) == 0:
            return None
        elif len(many_types) == 1:
            return many_types[0]
        dtype1, dtype2, *many_types = many_types
        if len(many_types) > 0:
            return upcast(upcast(dtype1, dtype2), *many_types)
        # here, we only have torch dtypes
        if dtype1 is None:
            return dtype2
        elif dtype2 is None:
            return dtype1
        elif dtype1 is torch.complex128 or dtype2 is torch.complex128:
            return torch.complex128
        elif dtype1 is torch.complex64 or dtype2 is torch.complex64:
            return torch.complex64
        elif hasattr(torch, 'complex32') and (dtype1 is torch.complex32 or
                                              dtype2 is torch.complex32):
            return torch.complex32
        elif dtype1 is torch.float64 or dtype2 is torch.float64:
            return torch.float64
        elif dtype1 is torch.float32 or dtype2 is torch.float32:
            return torch.float32
        elif dtype1 is torch.float16 or dtype2 is torch.float16:
            return torch.float16
        elif dtype1 is torch.int64 or dtype2 is torch.int64:
            return torch.int64
        elif dtype1 is torch.int32 or dtype2 is torch.int32:
            return torch.int32
        elif dtype1 is torch.int16 or dtype2 is torch.int16:
            return torch.int16
        elif dtype1 is torch.int8 and dtype2 is torch.int8:
            return torch.int8
        elif dtype1 is torch.uint8 and dtype2 is torch.uint8:
            return torch.uint8
        elif (dtype1 is torch.int8 and dtype2 is torch.uint8) or \
             (dtype1 is torch.uint8 and dtype2 is torch.int8):
            return torch.int16
        elif dtype1 is torch.bool and dtype2 is torch.bool:
            return torch.bool
        else:
            raise TypeError('We do not deal with type {} or {} yet.'
                            .format(dtype1, dtype2))

    def explore_dtype(x, n_pass=1):
        # find the max data type at a given pass
        if x is None:
            return None
        elif is_dtype(x):
            return dtypes.as_torch(x)
        elif (is_tensor(x) or is_array(x)) and len(x.shape) > 0:
            return dtypes.as_torch(x.dtype)
        elif is_tensor(x) or is_array(x):
            # scalar: only return if pass 2+
            return dtypes.as_torch(x.dtype) if n_pass >= 2 else None
        elif isinstance(x, numbers.Number):
            # builtin type:  only return if pass 3+
            return dtypes.as_torch(type(x)) if n_pass >= 3 else None
        else:
            # assume it is a sequence: check what we find in there
            return upcast(*[explore_dtype(elem, n_pass) for elem in x])

    # 1) tensors/arrays with dim > 0
    maxdtype = explore_dtype(args, n_pass=1)

    # 2) tensor/arrays with dim == 0
    if maxdtype is None:
        maxdtype = upcast(maxdtype, explore_dtype(args, n_pass=2))

    # 3) tensor/arrays
    if maxdtype is None:
        maxdtype = upcast(maxdtype, explore_dtype(args, n_pass=3))

    # Finally) ensure float
    if force_float:
        maxdtype = upcast(maxdtype, torch.get_default_dtype())

    return maxdtype


def same_storage(x, y):
    # type: (torch.Tensor, torch.Tensor) -> bool
    """Return true if `x` and `y` share the same underlying storage."""
    return x.storage().data_ptr() == y.storage().data_ptr()


def all_resident_tensors(no_duplicates=True):
    """Return all tensors currently allocated"""
    import gc
    objs = []

    def already_in(x):
        for i, obj in enumerate(objs):
            if same_storage(obj, x):
                if x.numel() > obj.numel():
                    del objs[i]
                    return False
                return True
        return False

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not (no_duplicates and already_in(obj)):
                    objs.append(obj)
                if (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    if not already_in(obj.data):
                        objs.append(obj.data)
        except:
            pass

    return objs


def print_all_resident_tensors(no_duplicates=True):
    """Print all tensors currently allocated"""
    for obj in all_resident_tensors(no_duplicates):
        print(type(obj), obj.shape)




def requires_grad(ctx, name):
    """Checks if a named variable requires gradients."""
    for g, n in zip(ctx.needs_input_grad, ctx.names):
        if n == name:
            return g
    return False


def fast_slice_tensor(x, index, dim=-1):
    """Index a tensor along one dimensions.

    This function is relatively similar to `torch.index_select`, except
    that it uses the native indexing mechanism and can therefore
    returns a tensor that use the same storage as the input tensor.

    It is faster but less versatile than `slice_tensor`.

    Parameters
    ----------
    x : tensor
        Input tensor.
    index : int or list[int] or slice
        Indices to select along `dim`.
    dim : int, default=last
        Dimension to index.

    Returns
    -------
    y : tensor
        Output tensor.

    """
    slicer = [slice(None)] * x.dim()
    slicer[dim] = index
    slicer = tuple(slicer)
    return x[slicer]


def slice_tensor(x, index, dim=None):
    """Index a tensor along one or several dimensions.

    This function is relatively similar to `torch.index_select`, except
    that it uses the native indexing mechanism and can therefore
    returns a tensor that use the same storage as the input tensor.

    Parameters
    ----------
    x : tensor
        Input tensor.
    index : index_like or tuple[index_like]
        Indices to select along each dimension in `dim`.
        If multiple dimensions are indexed, they *must* be held in a
        tuple (not a list). Each index can be a long, list of long,
        slice or tensor of long, but *cannot* be an ellipsis or
        tensor of bool.
    dim : int or sequence[int], optional
        Dimensions to index. If it is a list, `index` *must* be a tuple.
        By default, the last `n` dimensions (where `n` is the number of
        indices in `index`) are used.


    Returns
    -------
    y : tensor
        Output tensor.

    """
    # format (dim, index) as (list, tuple) with same length
    if not isinstance(index, tuple):
        index = (index,)
    if dim is None:
        dim = list(range(-len(index), 0))
    dim = py.ensure_list(dim)
    nb_dim = max(len(index), len(dim))
    dim = py.ensure_list(dim, nb_dim)
    index = tuple(py.ensure_list(index, nb_dim))

    # build index
    full_index = [slice(None)] * x.dim()
    for d, ind in zip(dim, index):
        if ind is Ellipsis or (torch.is_tensor(ind) and
                               ind.dtype == torch.bool):
            raise TypeError('`index` cannot be an ellipsis or mask')
        full_index[d] = ind
    full_index = tuple(full_index)

    return x.__getitem__(full_index)


def roll(inp, shifts=1, dims=None, bound='dft'):
    r"""Like torch.roll, but with any boundary condition

    /!\ When dims is None, we do not flatten but shift all dimensions.
    /!\ This differs from the behavior of torch.roll .

    Parameters
    ----------
    inp : tensor
        Input
    shifts : [sequence of] int
        Amount by which to roll.
        Positive shifts to the right, negative to the left.
    dims : [sequence of] int
        Dimensions to roll.
        By default, shifts apply to all dimensions if a scalar,
        or to the last N if a sequence.
    bound : bound-like
        Boundary condition

    Returns
    -------
    out : tensor
        Rolled tensor

    """
    if dims is None:
        if isinstance(shifts, int):
            dims = list(range(inp.dim()))
        else:
            shifts = py.make_list(shifts)
            dims = list(range(-len(shifts), 0))
    dims = py.make_list(dims)
    shifts = py.make_list(shifts, len(dims))
    bound = map(bounds.to_nitorch, py.make_list(bound, len(dims)))
    bound = [getattr(bounds, b + '_') for b in bound]

    grid = [torch.arange(n, device=inp.device) for n in inp.shape]
    mult = [1] * inp.dim()
    for d, s, b in zip(dims, shifts, bound):
        grid[d] -= s
        grid[d], mult[d] = b(grid[d], inp.shape[d])
    grid = list(meshgrid_ij(*grid))
    if any(map(torch.is_tensor, mult)):
        mult = meshgrid_ij(*mult)
    mult = py.prod(mult)
    grid = jit.sub2ind_list(grid, inp.shape)

    out = inp.flatten()[grid]
    out *= mult
    return out


def channel2last(tensor):
    """Warps: Channel to Last dimension order.

    . Channel ordering is: (Batch, Channel, X, Y, Z)
    . Last ordering is: (Batch, X, Y, Z, Channel))

    /!\ This function changes the *shape* of the tensor but
    does not change its *memory layout* (no reallocation).
    """
    tensor = torch.as_tensor(tensor)
    tensor = tensor.permute((0,) + tuple(range(2, tensor.dim())) + (1,))
    return tensor


def last2channel(tensor):
    """Warps: Last to Channel dimension order.

    . Channel ordering is: (Batch, Channel, X, Y, Z)
    . Last ordering is: (Batch, X, Y, Z, Channel))

    /!\ This function changes the *shape* of the tensor but
    does not change its *memory layout* (no reallocation).
    """
    tensor = torch.as_tensor(tensor)
    tensor = tensor.permute((0, - 1) + tuple(range(1, tensor.dim()-1)))
    return tensor


def ensure_channel_last(x, dim=1):
    """Ensure that the channel dimension is the most rapidly changing one

    /!\ This function does not change the *shape* of the tensor but
    may change its *memory layout*.

    Parameters
    ----------
    x : tensor
        Input tensor
    dim : int, default=1
        Index of the channel dimension

    Returns
    -------
    x : tensor
        Tensor where dimension `dim` has stride 1

    """
    strides = x.strides()
    if strides[dim] != 1:
        x = movedim(x, dim, -1)
        x = x.to(memory_format=torch.contiguous_format)
        x = movedim(x, -1, dim)
    return x


def ensure_contiguous(x):
    """Ensure that the tensor is contiguous, with the most rapidly
    changing dimension on the right.

    /!\ This function does not change the *shape* of the tensor but
    may change its *memory layout*.

    Parameters
    ----------
    x : tensor
        Input tensor

    Returns
    -------
    x : tensor
        Tensor with a contiguous layout

    """
    return x.to(memory_format=torch.contiguous_format)


def isin(tensor, labels):
    """Returns a mask for elements that belong to labels

    Parameters
    ----------
    tensor : (*shape_tensor) tensor_like
        Input tensor
    labels : (*shape_labels, nb_labels) tensor_like
        Labels.
        `shape_labels` and `shape_tensor` should be broadcastable.

    Returns
    -------
    mask : (*shape) tensor[bool]

    """

    tensor = torch.as_tensor(tensor)
    if isinstance(labels, set):
        labels = list(labels)
    labels = torch.as_tensor(labels)

    if labels.shape[-1] == 1:
        # only one label in the list
        return tensor == labels[..., 0]

    mask = tensor.new_zeros(tensor.shape, dtype=torch.bool)
    for label in torch.unbind(labels, dim=-1):
        mask = mask | (tensor == label)

    return mask


def ceil_pow(t, p=2.0, l=2.0, mx=None):
    """Ceils each element in vector t to the
    closest n that satisfies: l*p**n.

    This function is useful, for example, to ensure an image's dimensions
    work well in an encoding/decoding architecture.

    Parameters
    ----------
    t : (d, ), tensor
    p : float, default=2.0
    l : float, default=2.0
    mx : float, optional

    Returns
    ----------
    ct : (d, ), tensor

    """
    ct = t.clone()  # Do not modify in-place
    device = ct.device
    dtype0 = ct.dtype
    dtype = torch.float32
    dim = torch.as_tensor(ct, dtype=dtype, device=device)
    ct.clamp_max_(mx)
    d = len(ct)
    # Build array of l*p**[0, ..., N]
    N = 32
    p = torch.tensor(l, dtype=dtype, device=device) \
        * torch.tensor(p, dtype=dtype, device=device) \
        ** torch.arange(0, N, dtype=dtype, device=device)
    p = p.repeat(d, 1)
    # Ensure we ceil
    for n in range(d):
        p[n, p[n, ...] < ct[n]] = -inf
    ct = ct[..., None]
    # Find closest indices
    ix = torch.min((p - ct).abs(), dim=1)[1]
    ct = ct.squeeze()
    # Ceil input
    for n in range(d):
        if torch.isfinite(p[n, ix[n]]):
            ct[n] = p[n, ix[n]]
    # Return same datatype
    ct = ct.type(dtype0)

    return ct


def sub2ind(subs, shape, out=None):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D, ...) tensor_like
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) vector_like
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    ind : (...) tensor
        Linear indices
    """
    *subs, ind = subs
    if out is None:
        ind = torch.as_tensor(ind).clone()
    else:
        out.reshape(ind.shape).copy_(ind)
        ind = out
    bck = backend(ind)
    stride = py.cumprod(shape[1:], reverse=True)
    for i, s in zip(subs, stride):
        ind += torch.as_tensor(i, **bck) * torch.as_tensor(s, **bck)
    return ind


# floor_divide returns wrong results for negative values, because it truncates
# instead of performing a proper floor. In recent version of pytorch, it is
# advised to use div(..., rounding_mode='trunc'|'floor') instead.
# Here, we only use floor_divide on positive values so we do not care.
_trunc_div = ((lambda *a, **k: torch.div(*a, **k, rounding_mode='trunc'))
              if torch_version('>=', (1, 8)) else torch.floor_divide
              if torch_version('>=', (1, 5)) else (lambda x, y, **k: x // y))


def ind2sub(ind, shape, out=None):
    """Convert linear indices into sub indices (i, j, k).

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    ind : tensor_like
        Linear indices
    shape : (D,) vector_like
        Size of each dimension.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    subs : (D, ...) tensor
        Sub-indices.
    """
    ind = torch.as_tensor(ind)
    bck = backend(ind)
    stride = py.cumprod(shape, reverse=True, exclusive=True)
    stride = torch.as_tensor(stride, **bck)
    if out is None:
        sub = ind.new_empty([len(shape), *ind.shape])
    else:
        sub = out.reshape([len(shape), *ind.shape])
    sub[:, ...] = ind
    for d in range(len(shape)):
        if d > 0:
            torch.remainder(sub[d], torch.as_tensor(stride[d-1], **bck), out=sub[d])
        sub[d] = _trunc_div(sub[d], stride[d], out=sub[d])
    return sub


def _one_hot_wrapper(x: Tensor, dtype: Optional[torch.dtype] = None):
    x = x.long()
    x = torch.nn.functional.one_hot(x)
    x = x.to(dtype)
    return x


if torch_version('>=', (1, 6)):
    # jit.script does not accept `dtype` inputs in torch 1.3
    # I don't know exactly which version started handling it.
    _one_hot_wrapper = torch.jit.script(_one_hot_wrapper)


def one_hot(x, dim=-1, exclude_labels=None, exclude_missing=False, max_label=None,
            implicit=False, implicit_index=0, dtype=None, return_lookup=False):
    """One-hot encode a volume of labels.

    Parameters
    ----------
    x : tensor
        An integer-type tensor with label values.
    dim : int, default=-1
        Dimension in which to insert the one-hot channel.
    exclude_labels : sequence[int], optional
        A list of labels to exclude from one-hot encoding.
    exclude_missing : bool, default=False
        Exclude missing labels from one-hot encoding
        (their channel will be squeezed)
    max_label : int, optional
        Maximum label value
    implicit : bool, default=False
        Make the returned tensor have an implicit background class.
        In this case, output probabilities do not sum to one, but to some
        value smaller than one.
    implicit_index : int, default=-1
        Output channel to make implicit
    dtype : tensor.dtype, optional
        Output data type.
    return_lookup : bool, default=False
        Return lookup table from one-hot indices to labels

    Returns
    -------
    y : tensor
        One-hot tensor.
        The number of one-hot channels is equal to `x.max() - len(exclude) + 1`
        if not `implicit` else `x.max() - len(exclude)`.

    """
    if not exclude_labels and not exclude_missing and not implicit and not max_label:
        x = _one_hot_wrapper(x, dtype)
        x = fast_movedim(x, -1, dim)
        return x

    nb_classes = (max_label or int(x.max().item())) + 1
    exclude_labels = set(py.ensure_list(exclude_labels or []))
    if exclude_missing:
        all_labels = x.unique()
        missing_labels = [i for i in range(nb_classes) if i not in all_labels]
        exclude_labels = exclude_labels.union(missing_labels)

    dtype = dtype or x.dtype
    out = torch.zeros([nb_classes-implicit, *x.shape], dtype=dtype, device=x.device)
    implicit_index = (nb_classes + implicit_index if implicit_index < 0 else
                      implicit_index)
    i = 0
    lookup = []
    for j in range(nb_classes):
        if j in exclude_labels:
            continue
        if i == implicit_index:
            implicit_index = None
            continue
        out[i] = (x == j)
        lookup.append(j)
        i += 1

    out = fast_movedim(out, 0, dim)
    return (out, lookup) if return_lookup else out


def merge_labels(x, lookup):
    """Relabel a label tensor according to a lookup table

    Parameters
    ----------
    x : tensor
    lookup : sequence of [sequence of] int

    Returns
    -------
    x : tensor

    """
    out = torch.zeros_like(x)
    for i, j in enumerate(lookup):
        j = py.make_list(j)
        out[isin(x, j)] = i
    return out


def min_intensity_step(x, max_points=1e6):
    """Detect empty steps in the data distribution"""
    x = x.flatten()
    nb_points = x.numel()
    ratio = 1/min(1, max_points / nb_points)
    mask = torch.rand_like(x) > (1 - ratio)
    x = x[mask]
    x = x[torch.isfinite(x)]
    x = x.sort().values
    x = x[1:] - x[:-1]
    x = x[x > 0].min().item()
    return x


if torch_version('>=', (1, 10)):
    @torch.jit.script
    def meshgrid_script_ij(x: List[torch.Tensor]) -> List[Tensor]:
        return torch.meshgrid(x, indexing='ij')
    @torch.jit.script
    def meshgrid_script_xy(x: List[torch.Tensor]) -> List[Tensor]:
        return torch.meshgrid(x, indexing='xy')
    meshgrid_ij = lambda *x: torch.meshgrid(*x, indexing='ij')
    meshgrid_xy = lambda *x: torch.meshgrid(*x, indexing='xy')
else:
    @torch.jit.script
    def meshgrid_script_ij(x: List[torch.Tensor]) -> List[Tensor]:
        return torch.meshgrid(x)
    @torch.jit.script
    def meshgrid_script_xy(x: List[torch.Tensor]) -> List[Tensor]:
        grid = torch.meshgrid(x)
        if len(grid) > 1:
            grid[0] = grid[0].transpose(0, 1)
            grid[1] = grid[1].transpose(0, 1)
        return grid
    meshgrid_ij = lambda *x: torch.meshgrid(*x)
    def meshgrid_xy(*x):
        grid = list(torch.meshgrid(*x))
        if len(grid) > 1:
            grid[0] = grid[0].transpose(0, 1)
            grid[1] = grid[1].transpose(0, 1)
        return grid


