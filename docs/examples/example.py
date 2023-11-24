"""Nice module."""
from __future__ import annotations

from epseon_backend.device.gpu._libepseon_gpu import (
    EpseonComputeContext,
    MorsePotentialConfig,
)

ctx = EpseonComputeContext.create()


device_info = next(iter(ctx.get_physical_device_info()))
interface = ctx.get_device_interface(device_info.device_properties.device_id)
configurator = interface.get_task_configurator("float32")
cfg = (
    configurator.set_hardware_config(
        potential_buffer_size=16500,
        group_size=512,
        allocation_block_size=16 * 1024 * 1024,
    )
    .set_morse_potential(
        [
            MorsePotentialConfig(
                dissociation_energy=500,
                equilibrium_bond_distance=2.6,
                well_width=1.3,
                min_r=0.0,
                max_r=10.0,
                point_count=16500,
            ),
            MorsePotentialConfig(
                dissociation_energy=5500.0,
                equilibrium_bond_distance=0.6,
                well_width=10,
                min_r=0.0,
                max_r=10.0,
                point_count=16500,
            ),
        ],
    )
    .set_vibwa_algorithm(
        mass_atom_0=87.62,
        mass_atom_1=87.62,
        integration_step=0.1,
        min_distance_to_asymptote=0.1,
        min_level=0,
        max_level=0,
    )
)

task_handle = interface.submit_task(cfg)

task_handle.get_status_message()
task_handle.is_done()
task_handle.wait()
