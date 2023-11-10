#pragma once

#include "epseon/libepseon.hpp"
#include "epseon_gpu/vulkan_application.hpp"
#include <string>

namespace epseon {
    namespace gpu {
        namespace python {
            /* Python API function - returns Vulkan version extracted from
             * VkInstance. */
            std::string                          get_vulkan_version();
            std::vector<cpp::PhysicalDeviceInfo> get_physical_device_info();
        } // namespace python
    }     // namespace gpu
} // namespace epseon
