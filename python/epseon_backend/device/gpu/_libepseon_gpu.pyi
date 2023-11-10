from __future__ import annotations

from typing import Iterable, Literal, Protocol

def get_vulkan_version() -> str:
    """Get Vulkan API version."""

class PhysicalDeviceLimits(Protocol):
    """Physical device limits - mostly max counts of different resources."""

    max_uniform_buffer_range: int
    max_storage_buffer_range: int
    max_push_constants_size: int
    max_memory_allocation_count: int
    sparse_address_space_size: int
    max_bound_descriptor_sets: int
    max_per_stage_descriptor_samplers: int
    max_per_stage_descriptor_uniform_buffers: int
    max_per_stage_descriptor_storage_buffers: int
    max_per_stage_descriptor_sampled_images: int
    max_per_stage_descriptor_storage_images: int
    max_per_stage_descriptor_input_attachments: int
    max_per_stage_resources: int
    max_descriptor_set_samplers: int
    max_descriptor_set_uniform_buffers: int
    max_descriptor_set_uniform_buffers_dynamic: int
    max_descriptor_set_storage_buffers: int
    max_descriptor_set_storage_buffers_dynamic: int
    max_descriptor_set_sampled_images: int
    max_descriptor_set_storage_images: int
    max_descriptor_set_input_attachments: int

    max_compute_shared_memory_size: int
    max_compute_work_group_count: tuple[int, int, int]
    max_compute_work_group_invocations: int
    max_compute_work_group_size: int

    timestamp_compute_and_graphics: int
    timestamp_period: int

class PhysicalDeviceProperties(Protocol):
    """Properties of physical device retrieved from Vulkan API."""

    api_version: str
    driver_version: str
    vendor_id: int
    device_id: int
    device_type: Literal[
        "integrated_gpu",
        "discrete_gpu",
        "virtual_gpu",
        "cpu",
        "other",
    ]
    device_name: str
    pipeline_cache_uuid: list[int]
    limits: PhysicalDeviceLimits
    sparse_properties: str

class PhysicalDeviceInfo(Protocol):
    """Container for physical device info retrieved from Vulkan API."""

    device_properties: PhysicalDeviceProperties

def get_physical_device_info() -> Iterable[PhysicalDeviceInfo]:
    """Get information about available physical devices."""
