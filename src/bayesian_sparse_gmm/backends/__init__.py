import warnings

from ._base import ComputeBackend
from ._numpy import NumpyBackend

try:
    from ._numba import NumbaBackend

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ._cuda import CUPY_AVAILABLE, CUDABackend


def select_backend(preference: str = "auto") -> ComputeBackend:
    """Select the best available compute backend.

    Parameters
    ----------
    preference : str
        One of 'numpy', 'numba', 'cuda', or 'auto'.
        - 'cuda': CuPy GPU → Numba CUDA GPU → fallback to NumPy.
        - 'numba': Numba CPU parallel → fallback to NumPy.
        - 'auto': Numba CPU parallel → NumPy.
        - 'numpy': Pure NumPy (always available).
    """
    preference = preference.lower()

    if preference == "numpy":
        return NumpyBackend()

    if preference == "cuda":
        if CUPY_AVAILABLE:
            return CUDABackend()
        if NUMBA_AVAILABLE:
            return NumbaBackend(use_cuda=True)
        warnings.warn(
            "CUDA requested but neither CuPy nor Numba CUDA is available. "
            "Falling back to NumPy.",
            stacklevel=2,
        )
        return NumpyBackend()

    if preference == "numba":
        if NUMBA_AVAILABLE:
            return NumbaBackend(use_cuda=False)
        warnings.warn(
            "Numba requested but not available. Falling back to NumPy.",
            stacklevel=2,
        )
        return NumpyBackend()

    if preference == "auto":
        if NUMBA_AVAILABLE:
            return NumbaBackend(use_cuda=False)
        return NumpyBackend()

    raise ValueError(f"Unknown backend: {preference!r}")
