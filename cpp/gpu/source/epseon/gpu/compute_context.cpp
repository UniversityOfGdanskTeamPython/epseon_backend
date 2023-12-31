#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/common.hpp"
#include "epseon/gpu/compute_context.hpp"
#include "epseon/gpu/device_interface.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace epseon::gpu::cpp {

    ComputeContextState::ComputeContextState(ComputeContextState& ccs) = default;

    ComputeContextState::ComputeContextState(
        std::shared_ptr<spdlog::logger>      logger_,
        std::shared_ptr<vk::raii::Context>   context_,
        std::shared_ptr<vk::ApplicationInfo> application_info_,
        std::shared_ptr<vk::raii::Instance>  instance_
    ) :
        logger(std::move(logger_)),
        context(std::move(context_)),
        application_info(std::move(application_info_)),
        instance(std::move(instance_)) {}

    ComputeContext::ComputeContext(
        std::shared_ptr<spdlog::logger>      logger_,
        std::shared_ptr<vk::raii::Context>   context_,
        std::shared_ptr<vk::ApplicationInfo> application_info_,
        std::shared_ptr<vk::raii::Instance>  instance_
    ) :
        state(std::make_shared<ComputeContextState>(logger_, context_, application_info_, instance_)
        ) {}

    std::shared_ptr<ComputeContext> ComputeContext::create(uint32_t version) {
        auto logger = spdlog::get("_libepseon_gpu");
        if (!logger) {
            logger = spdlog::basic_logger_mt("_libepseon_gpu", "./log/epseon/gpu/log.txt");
        }

        auto context = std::make_shared<vk::raii::Context>();

        uint32_t apiVersion = context->enumerateInstanceVersion();
        if (apiVersion < VK_API_VERSION_1_3) {
            logger->warn(
                "Unsupported Vulkan version: {}", common::vulkan_version_to_string(apiVersion)
            );
        } else if (apiVersion < VK_API_VERSION_1_1) {
            auto str = fmt::format(
                "Insufficient Vulkan version {} try updating your drivers or "
                "SDK.",
                common::vulkan_version_to_string(apiVersion)
            );
            logger->critical(str);

            throw std::runtime_error(str);
        } else {
            logger->info(
                "Discovered Vulkan version: {}", common::vulkan_version_to_string(apiVersion)
            );
        }

        auto applicationInfo = std::make_shared<vk::ApplicationInfo>();
        applicationInfo->setApplicationVersion(version)
            .setPApplicationName("libepseon_gpu")
            .setEngineVersion(version)
            .setPEngineName("libepseon_gpu")
            .setApiVersion(VK_API_VERSION_1_1);

        std::vector<const char*> instanceExtensions{};

        auto instanceCreateInfo = vk::InstanceCreateInfo()
                                      .setEnabledExtensionCount(instanceExtensions.size())
                                      .setPEnabledExtensionNames(instanceExtensions)
                                      .setPApplicationInfo(applicationInfo.get());
        auto instance = std::make_shared<vk::raii::Instance>(
            std::move(context->createInstance(instanceCreateInfo))
        );

        return std::make_shared<ComputeContext>(logger, context, applicationInfo, instance);
    }

    std::string ComputeContext::getVulkanAPIVersion() {
        auto apiVersion = this->state->context->enumerateInstanceVersion();
        return common::vulkan_version_to_string(apiVersion);
    }

    std::vector<PhysicalDeviceInfo> ComputeContext::getPhysicalDevicesInfo() {
        std::vector<PhysicalDeviceInfo> devices;

        for (const auto& physicalDevice : this->state->instance->enumeratePhysicalDevices()) {
            auto deviceProperties = physicalDevice.getProperties();
            auto memoryProperties = physicalDevice.getMemoryProperties();
            devices.push_back({deviceProperties, memoryProperties});
        }

        return devices;
    }

    std::shared_ptr<ComputeDeviceInterface> ComputeContext::getDeviceInterface(uint32_t deviceId) {
        for (auto& physicalDevice : vk::raii::PhysicalDevices{this->state->getVkInstance()}) {
            auto props = physicalDevice.getProperties();

            if (props.deviceID == deviceId) {
                auto physicalDevicePtr =
                    std::make_shared<vk::raii::PhysicalDevice>(std::move(physicalDevice));
                return std::make_shared<ComputeDeviceInterface>(this->state, physicalDevicePtr);
            }
        }
        throw std::runtime_error("Device not available.");
    };
} // namespace epseon::gpu::cpp
