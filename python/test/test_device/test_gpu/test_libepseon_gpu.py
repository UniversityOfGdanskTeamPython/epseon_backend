"""Test components of `epseon_backend.device.gpu._libepseon_gpu` submodule."""
from __future__ import annotations


def test_greet() -> None:
    """Check if temporary `greet()` method exported from _libepseon_gpu is available."""
    from epseon_backend.device.gpu._libepseon_gpu import greet

    greet()
