"""Test components of `epseon_backend.device.gpu._libepseon_gpu` submodule."""
from __future__ import annotations

import logging
import re

from epseon_backend.format import convert_size_in_bytes_to_adaptive_unit

COMPUTE_GROUP_AXES_COUNT = 3


class TestEpseonComputeContext:
    """Test EpseonComputeContext capabilities."""

    def test_create_epseon_compute_context(self) -> None:
        """Check if it is possible to instantiate EpseonComputeContext."""
        from epseon_backend.device.gpu._libepseon_gpu import EpseonComputeContext

        context = EpseonComputeContext.create()
        assert repr(context)

    def test_get_vulkan_version(self) -> None:
        """Check if temporary `greet()` method exported from _libepseon_gpu is available."""
        from epseon_backend.device.gpu._libepseon_gpu import EpseonComputeContext

        ctx = EpseonComputeContext.create()

        assert re.match(r"\d+\.\d+\.\d+\.\d+", ctx.get_vulkan_version()) is not None

    def test_get_physical_device_info_device_properties(self) -> None:
        """Check if getting information about available physical devices is possible."""
        from epseon_backend.device.gpu._libepseon_gpu import EpseonComputeContext

        ctx = EpseonComputeContext.create()

        assert len(tuple(ctx.get_physical_device_info())) > 0

        for device in ctx.get_physical_device_info():
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
                "INTEGRATED_GPU",
                "DISCRETE_GPU",
                "VIRTUAL_GPU",
                "CPU",
                "OTHER",
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

    def test_get_physical_device_info_memory_properties(self) -> None:
        """Check if getting information about available physical devices is possible."""
        from epseon_backend.device.gpu._libepseon_gpu import EpseonComputeContext

        ctx = EpseonComputeContext.create()

        assert len(tuple(ctx.get_physical_device_info())) > 0

        for device in ctx.get_physical_device_info():
            logging.info(
                "%s %s %s",
                device.device_properties.device_name,
                device.device_properties.api_version,
                device.device_properties.driver_version,
            )
            assert hasattr(device, "memory_properties")
            assert hasattr(device.memory_properties, "memory_heaps")

            logging.info("MEMORY HEAPS")
            for i, heap in enumerate(device.memory_properties.memory_heaps):
                logging.info("- Memory heap #%s", i)
                logging.info("  - heap flags = %s", heap.flags)
                logging.info(
                    "  - heap size = %s",
                    convert_size_in_bytes_to_adaptive_unit(heap.size),
                )

            logging.info("MEMORY TYPES")
            for i, mem_type in enumerate(device.memory_properties.memory_types):
                logging.info("- Memory type #%s", i)
                logging.info("  - type flags = %s", mem_type.flags)
                logging.info(
                    "  - type heap index = %s",
                    mem_type.heap_index,
                )

    def test_get_device_interface(self) -> None:
        """Check if getting information about available physical devices is possible."""
        from epseon_backend.device.gpu._libepseon_gpu import EpseonComputeContext

        ctx = EpseonComputeContext.create()

        device_info = next(iter(ctx.get_physical_device_info()))
        interface = ctx.get_device_interface(device_info.device_properties.device_id)
        assert id(interface)
