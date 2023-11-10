"""Test components of `epseon_backend.device.gpu._libepseon_gpu` submodule."""
from __future__ import annotations

import logging
import re

COMPUTE_GROUP_AXES_COUNT = 3


def test_get_vulkan_version() -> None:
    """Check if temporary `greet()` method exported from _libepseon_gpu is available."""
    from epseon_backend.device.gpu._libepseon_gpu import get_vulkan_version

    assert re.match(r"\d+\.\d+\.\d+\.\d+", get_vulkan_version()) is not None


def test_get_physical_device_info() -> None:
    """Check if getting information about available physical devices is possible."""
    from epseon_backend.device.gpu._libepseon_gpu import get_physical_device_info

    assert len(tuple(get_physical_device_info())) > 0

    for device in get_physical_device_info():
        logging.info(
            "%s %s %s",
            device.device_properties.device_name,
            device.device_properties.api_version,
            device.device_properties.driver_version,
        )
        assert isinstance(device.device_properties.api_version, str)
        assert isinstance(device.device_properties.driver_version, str)
        assert isinstance(device.device_properties.vendor_id, int)
        assert isinstance(device.device_properties.device_id, int)
        assert device.device_properties.device_type in (
            "integrated_gpu",
            "discrete_gpu",
            "virtual_gpu",
            "cpu",
            "other",
        )
        assert isinstance(device.device_properties.device_name, str)
        assert len(device.device_properties.device_name) > 0
        assert isinstance(device.device_properties.pipeline_cache_uuid, list)
        assert isinstance(
            device.device_properties.limits.max_compute_shared_memory_size,
            int,
        )
        assert isinstance(
            device.device_properties.limits.max_compute_work_group_count,
            tuple,
        )
        assert (
            len(device.device_properties.limits.max_compute_work_group_count)
            == COMPUTE_GROUP_AXES_COUNT
        )
        assert isinstance(
            device.device_properties.limits.max_compute_work_group_invocations,
            int,
        )
        assert isinstance(
            device.device_properties.limits.max_compute_work_group_size,
            int,
        )
        assert hasattr(
            device.device_properties,
            "sparse_properties",
        )
