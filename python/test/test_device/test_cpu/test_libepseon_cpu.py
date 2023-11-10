"""Test components of `epseon_backend.device.cpu._libepseon_cpu` submodule."""
from __future__ import annotations


def test_greet() -> None:
    """Check if temporary `greet()` method exported from _libepseon_cpu is available."""
    from epseon_backend.device.cpu._libepseon_cpu import greet

    greet()
