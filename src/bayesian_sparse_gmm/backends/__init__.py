import warnings

from ._base import ComputeBackend
from ._numpy import NumpyBackend

try:
    from ._numba import NumbaBackend

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ._cuda import CUPY_AVAILABLE, CUDABackend


def select_backend(preference: str = "auto", use_cuda: bool = False) -> ComputeBackend:
    """Select the best available compute backend based on preference."""
    preference = preference.lower()

    if preference == "numpy":
        return NumpyBackend()

    if preference == "cuda":
        if CUPY_AVAILABLE:
            return CUDABackend()
        if NUMBA_AVAILABLE:
            return NumbaBackend(use_cuda=True)
        warnings.warn(
            "CUDA backend requested but neither CuPy nor Numba CUDA is available. Falling back to NumPy."
        )
        return NumpyBackend()

    if preference in ("numba", "auto"):
        if preference == "numba":
            if NUMBA_AVAILABLE:
                return NumbaBackend(use_cuda=use_cuda)
            warnings.warn(
                "Numba backend requested but Numba is not available. Falling back to NumPy."
            )
            return NumpyBackend()

        # preference == "auto"
        if use_cuda:
            if CUPY_AVAILABLE:
                return CUDABackend()
            if NUMBA_AVAILABLE:
                return NumbaBackend(use_cuda=True)
        if NUMBA_AVAILABLE:
            return NumbaBackend(use_cuda=False)
        return NumpyBackend()

    raise ValueError(f"Unknown backend preference: {preference}")
