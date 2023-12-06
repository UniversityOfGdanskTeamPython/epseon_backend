#include "epseon/gpu/device_interface.hpp"
#include "epseon/gpu/task_configurator/task_configurator.hpp"
#include <memory>

#include "epseon/vulkan_headers.hpp"

namespace epseon {
    namespace gpu {
        namespace cpp {
            ComputeDeviceInterface::ComputeDeviceInterface(
                std::shared_ptr<ComputeContextState>      computeContextState_,
                std::shared_ptr<vk::raii::PhysicalDevice> physicalDevice_
            ) :
                computeContextState(computeContextState_),
                physicalDevice(physicalDevice_) {}

            const vk::raii::PhysicalDevice& ComputeDeviceInterface::getPhysicalDevice() const {
                return *this->physicalDevice;
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
