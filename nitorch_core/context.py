import torch
import numpy as np
import random
import os
import contextlib
from nitorch_core import py


class benchmark:
    """Context manager for the convolution benchmarking utility
    from pytorch.

    When the benchmark value is True, each time a convolution is called
    on a new input shape, several algorithms are performed and evaluated,
    and the best one kept in memory. Therefore, benchmarking is beneficial
    if and only if the (channel + spatial) shape of your input data is
    constant.

    Examples
    --------
    ```python
    from nitorch.core.utils import benchmark
    with benchmark(True):
        train_my_model(model)
    ```

    """

    def __init__(self, value=True):
        self.do_benchmark = value

    def __enter__(self):
        self.prev_value = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self.do_benchmark

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.backends.cudnn.benchmark = self.prev_value


@contextlib.contextmanager
def reproducible(seed=1234, enabled=True, devices=None):
    """Context manager for a reproducible environment

    Parameters
    ----------
    seed : int, default=1234
        Initial seed to use in all forked RNGs
    enabled : bool, default=True
        Set to False to disable the context manager
    devices : [list of] int or str or torch.device, optional
        CUDA devices. All devices are forked by default.

    Example
    -------
    ```python
    from nitorch.core.utils import reproducible
    with reproducible():
        train_my_model(model)
    ```

    """
    if not enabled:
        yield
        return

    # save initial states
    py_state = random.getstate()
    py_hash_seed = os.environ.get('PYTHONHASHSEED', None)
    cudnn = torch.backends.cudnn.deterministic
    cpu_state = torch.random.get_rng_state()
    devices = py.make_list(devices or [])
    devices = [torch.device(device) for device in devices
               if torch.device(device).type == 'cuda']
    cuda_state = [torch.cuda.get_rng_state(device) for device in devices]
    np_state = np.random.get_state() if np else None

    try:  # fork states
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.random.manual_seed(seed)
        if np:
            np.random.seed(seed)
        yield

    finally:   # reset initial states
        random.setstate(py_state)
        if 'PYTHONHASHSEED' in os.environ:
            if py_hash_seed is not None:
                os.environ['PYTHONHASHSEED'] = py_hash_seed
            else:
                del os.environ['PYTHONHASHSEED']
        torch.backends.cudnn.deterministic = cudnn
        torch.random.set_rng_state(cpu_state)
        for device, state in zip(devices, cuda_state):
            torch.cuda.set_rng_state(state, device)
        if np:
            np.random.set_state(np_state)


def manual_seed_all(seed=1234):
    """Set all possible random seeds to a fixed value"""
    if np:
        np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.random.manual_seed(seed)