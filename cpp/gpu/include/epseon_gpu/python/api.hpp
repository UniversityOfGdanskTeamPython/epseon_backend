#pragma once

#include "epseon/libepseon.hpp"
#include "epseon_gpu/compute_context.hpp"
#include "epseon_gpu/device_interface.hpp"
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace epseon {
    namespace gpu {
        namespace python {

            class ComputeDeviceInterface {
              public:
                std::shared_ptr<cpp::ComputeDeviceInterface> device;

              public:
                ComputeDeviceInterface(std::shared_ptr<cpp::ComputeDeviceInterface>);
            };

            class EpseonComputeContext {
              public:
                std::shared_ptr<cpp::ComputeContext> application = {};

              public:
                EpseonComputeContext(std::shared_ptr<cpp::ComputeContext>);

                static EpseonComputeContext create();

                /* Python API function - returns Vulkan version extracted from
                 * VkInstance. */
                std::string                          get_vulkan_version();
                /* Python API function - Get information about available physical
                 * devices. */
                std::vector<cpp::PhysicalDeviceInfo> get_physical_device_info();
                /* Python API function - Get interface for running algorithms on Vulkan
                 * devices. */
                ComputeDeviceInterface               get_device_interface(uint32_t);
            };
        } // namespace python
    }     // namespace gpu
} // namespace epseon
