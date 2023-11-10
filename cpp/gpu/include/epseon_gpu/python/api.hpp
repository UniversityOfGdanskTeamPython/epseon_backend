#pragma once

#include "epseon/libepseon.hpp"
#include "epseon_gpu/vulkan_application.hpp"
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace epseon {
    namespace gpu {
        namespace python {

            class EpseonComputeContext {
              public:
                std::unique_ptr<cpp::VulkanApplication> application = {};

              public:
                EpseonComputeContext(std::unique_ptr<cpp::VulkanApplication> application
                );

                static std::unique_ptr<EpseonComputeContext> create();

                /* Python API function - returns Vulkan version extracted from
                 * VkInstance. */
                std::string                          get_vulkan_version();
                /* Python API function - Get information about available physical
                 * devices. */
                std::vector<cpp::PhysicalDeviceInfo> get_physical_device_info();
            };
        } // namespace python
    }     // namespace gpu
} // namespace epseon
