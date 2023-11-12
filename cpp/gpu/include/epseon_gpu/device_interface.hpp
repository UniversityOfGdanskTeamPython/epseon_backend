#pragma once

#include "epseon_gpu/compute_context.hpp"
#include <memory>
#include <vulkan/vulkan_raii.hpp>

namespace epseon {
    namespace gpu {
        namespace cpp {
            class ComputeDeviceInterface {
              private:
                std::shared_ptr<ComputeContextState> computeContextState;
                vk::raii::PhysicalDevice             physicalDevice;

              public:
                ComputeDeviceInterface(
                    std::shared_ptr<ComputeContextState>, vk::raii::PhysicalDevice
                );
            };
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
