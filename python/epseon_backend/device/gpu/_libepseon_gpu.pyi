from __future__ import annotations

from typing import Iterable, Literal, Protocol

class PhysicalDeviceSparseProperties(Protocol):
    """Sparse resources properties retrieved from Vulkan API."""

class PhysicalDeviceLimits(Protocol):
    """Physical device limits - mostly max counts of different resources."""

    # In the future -> max_uniform_buffer_range: int
    # In the future -> max_storage_buffer_range: int
    # In the future -> max_push_constants_size: int
    # In the future -> max_memory_allocation_count: int
    # In the future -> sparse_address_space_size: int
    # In the future -> max_bound_descriptor_sets: int
    # In the future -> max_per_stage_descriptor_samplers: int
    # In the future -> max_per_stage_descriptor_uniform_buffers: int
    # In the future -> max_per_stage_descriptor_storage_buffers: int
    # In the future -> max_per_stage_descriptor_sampled_images: int
    # In the future -> max_per_stage_descriptor_storage_images: int
    # In the future -> max_per_stage_descriptor_input_attachments: int
    # In the future -> max_per_stage_resources: int
    # In the future -> max_descriptor_set_samplers: int
    # In the future -> max_descriptor_set_uniform_buffers: int
    # In the future -> max_descriptor_set_uniform_buffers_dynamic: int
    # In the future -> max_descriptor_set_storage_buffers: int
    # In the future -> max_descriptor_set_storage_buffers_dynamic: int
    # In the future -> max_descriptor_set_sampled_images: int
    # In the future -> max_descriptor_set_storage_images: int
    # In the future -> max_descriptor_set_input_attachments: int

    max_compute_shared_memory_size: int
    max_compute_work_group_count: tuple[int, int, int]
    max_compute_work_group_invocations: int
    max_compute_work_group_size: int

class PhysicalDeviceProperties(Protocol):
    """Properties of physical device retrieved from Vulkan API."""

    api_version: str
    driver_version: str
    vendor_id: int
    device_id: int
    device_type: Literal[
        "INTEGRATED_GPU",
        "DISCRETE_GPU",
        "VIRTUAL_GPU",
        "CPU",
        "OTHER",
    ]
    device_name: str
    pipeline_cache_uuid: list[int]
    limits: PhysicalDeviceLimits
    sparse_properties: PhysicalDeviceSparseProperties

class MemoryHeap(Protocol):
    """Wrapper around vk::MemoryHeap object."""

    size: int
    flags: list[str]

class MemoryType(Protocol):
    """Wrapper around vk::MemoryType object."""

    heap_index: int
    flags: list[str]

class PhysicalDeviceMemoryProperties(Protocol):
    """Properties of physical device memory retrieved from Vulkan API."""

    memory_heaps: list[MemoryHeap]
    memory_types: list[MemoryType]

class PhysicalDeviceInfo(Protocol):
    """Container for physical device info retrieved from Vulkan API."""

    device_properties: PhysicalDeviceProperties
    memory_properties: PhysicalDeviceMemoryProperties

class ComputeDeviceInterface:
    """Interface to particular Vulkan device."""

class EpseonComputeContext(Protocol):
    """Interface to computations on GPU with Vulkan."""

    @staticmethod
    def create() -> EpseonComputeContext:
        """Create new instance of EpseonComputeContext type."""
    def get_vulkan_version(self) -> str:
        """Get Vulkan API version."""
    def get_physical_device_info(self) -> Iterable[PhysicalDeviceInfo]:
        """Get information about available physical devices."""
    def get_device_interface(self, __device_id: int) -> ComputeDeviceInterface:
        """Get interface for running algorithms on Vulkan devices."""
