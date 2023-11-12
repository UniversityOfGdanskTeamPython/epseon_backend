#pragma once

#include "epseon/libepseon.hpp"

#include "spdlog/logger.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace epseon {
    namespace gpu {
        namespace cpp {

            struct ComputeContextState {
                std::shared_ptr<spdlog::logger>      logger           = {};
                std::shared_ptr<vk::raii::Context>   context          = {};
                std::shared_ptr<vk::ApplicationInfo> application_info = {};
                std::shared_ptr<vk::raii::Instance>  instance         = {};

                ComputeContextState(ComputeContextState&);
                ComputeContextState(std::shared_ptr<spdlog::logger>, std::shared_ptr<vk::raii::Context>, std::shared_ptr<vk::ApplicationInfo>, std::shared_ptr<vk::raii::Instance>);
            };

            struct PhysicalDeviceInfo {
              public:
                vk::PhysicalDeviceProperties       deviceProperties;
                vk::PhysicalDeviceMemoryProperties memoryProperties;
            };

            class ComputeDeviceInterface;

            class ComputeContext {
              private:
                std::shared_ptr<ComputeContextState> state;

              public:
                ComputeContext(
                    std::shared_ptr<spdlog::logger>      logger_,
                    std::shared_ptr<vk::raii::Context>   context_,
                    std::shared_ptr<vk::ApplicationInfo> application_info_,
                    std::shared_ptr<vk::raii::Instance>  instance_
                );

              public:
                static std::shared_ptr<ComputeContext>
                create(uint32_t version = VK_MAKE_API_VERSION(0, 0, 1, 0));

              public:
                std::string                             getVulkanAPIVersion();
                std::vector<PhysicalDeviceInfo>         getPhysicalDevicesInfo();
                std::shared_ptr<ComputeDeviceInterface> getDeviceInterface(uint32_t);
            };
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
