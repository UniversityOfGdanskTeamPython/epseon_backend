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
            struct PhysicalDeviceInfo {
              public:
                vk::PhysicalDeviceProperties       deviceProperties;
                vk::PhysicalDeviceMemoryProperties memoryProperties;
            };

            class VulkanApplication {
              private:
                std::shared_ptr<spdlog::logger>      logger           = {};
                std::unique_ptr<vk::raii::Context>   context          = {};
                std::unique_ptr<vk::ApplicationInfo> application_info = {};
                std::unique_ptr<vk::raii::Instance>  instance         = {};

              private:
                VulkanApplication(
                    std::shared_ptr<spdlog::logger>      logger,
                    std::unique_ptr<vk::raii::Context>   context,
                    std::unique_ptr<vk::ApplicationInfo> application_info,
                    std::unique_ptr<vk::raii::Instance>  instance
                );

              public:
                static std::unique_ptr<VulkanApplication>
                create(uint32_t version = VK_MAKE_API_VERSION(0, 0, 1, 0));

              public:
                std::string                     getVulkanAPIVersion();
                std::vector<PhysicalDeviceInfo> getPhysicalDevicesInfo();
            };
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
