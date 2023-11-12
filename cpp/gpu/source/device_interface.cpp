#include "epseon_gpu/device_interface.hpp"
#include <vulkan/vulkan_raii.hpp>

namespace epseon {
    namespace gpu {
        namespace cpp {
            ComputeDeviceInterface::ComputeDeviceInterface(
                std::shared_ptr<ComputeContextState> computeContextState_,
                vk::raii::PhysicalDevice             physicalDevice_
            ) :
                computeContextState(computeContextState_),
                physicalDevice(physicalDevice_) {}
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
