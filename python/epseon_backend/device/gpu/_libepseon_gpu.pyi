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

class TaskConfigurator:
    """Builder for GPU compute task."""

    def set_hardware_config(
        self,
        potential_buffer_size: int,
        group_size: int,
        allocation_block_size: int,
    ) -> _PartialConfig1:
        """Set hardware configuration for GPU compute task."""

class MorsePotentialConfig:
    """Configuration of single Morse potential curve."""

    def __init__(  # noqa: PLR0913
        self,
        dissociation_energy: float,
        equilibrium_bond_distance: float,
        well_width: float,
        min_r: float,
        max_r: float,
        point_count: int,
    ) -> None:
        """Create instance of MorsePotentialConfig class."""

class _PartialConfig1:
    """Partially finished configuration on stage 1.

    Includes configuration for hardware.
    """

    def set_morse_potential(
        self,
        __configs: list[MorsePotentialConfig],
    ) -> _PartialConfig2:
        """Set potential data source configuration.

        Raises
        ------
        RuntimeError when MorsePotentialConfigs with different point counts are used.
        """

class _PartialConfig2:
    """Partially finished configuration on stage 2.

    Includes configuration for hardware and for potential source.
    """

    def set_vibwa_algorithm(
        self,
        integration_step: float,
        min_distance_to_asymptote: float,
        min_level: int,
        max_level: int,
    ) -> TaskConfig:
        """Set task algorithm configuration."""

class TaskConfig:
    """Finalized task configuration object."""

class TaskHandle:
    """Handle object for referencing GPU compute task."""

    def get_status_message(self) -> str:
        """Get task status message."""
    def is_done(self) -> bool:
        """Check if task has finished."""
    def wait(self) -> None:
        """Wait for task to finish."""

class ComputeDeviceInterface:
    """Interface to particular Vulkan device.

    ComputeDeviceInterface doesn't guard against submitting multiple tasks at a time,
    but doing so might result in undefined behavior in form of race conditions and
    memory occupancy problems, as it is not prepared for handling such case.
    """

    def get_task_configurator(
        self,
        __precision: Literal["float32", "float64"],
    ) -> TaskConfigurator:
        """Get new task configurator instance."""
    def submit_task(self, __config: TaskConfig) -> TaskHandle:
        """Submit task for execution."""

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
