
#include "epseon_gpu/vulkan_application.hpp"
#include "epseon_gpu/common.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "vulkan/vulkan_structs.hpp"
#include <iostream>
#include <string>

namespace epseon {
    namespace gpu {
        namespace cpp {

            VulkanApplication::VulkanApplication(
                std::shared_ptr<spdlog::logger>      logger,
                std::unique_ptr<vk::raii::Context>   context,
                std::unique_ptr<vk::ApplicationInfo> application_info,
                std::unique_ptr<vk::raii::Instance>  instance
            ) :
                logger(logger),
                context(std::move(context)),
                application_info(std::move(application_info)),
                instance(std::move(instance)) {}

            std::unique_ptr<VulkanApplication>
            VulkanApplication::create(uint32_t version) {
                auto logger = spdlog::get("_libepseon_gpu");
                if (!logger) {
                    logger = spdlog::basic_logger_mt(
                        "_libepseon_gpu", "./log/libepseon_gpu/log.txt"
                    );
                }

                auto context = std::make_unique<vk::raii::Context>();

                uint32_t apiVersion = context->enumerateInstanceVersion();
                if (apiVersion < VK_API_VERSION_1_3) {
                    logger->warn(
                        "Unsupported Vulkan version: {}",
                        common::vulkan_version_to_string(apiVersion)
                    );
                } else {
                    logger->info(
                        "Discovered Vulkan version: {}",
                        common::vulkan_version_to_string(apiVersion)
                    );
                }

                vk::apiVersionMajor(apiVersion);

                auto applicationInfo = std::make_unique<vk::ApplicationInfo>();
                applicationInfo->setApplicationVersion(version)
                    .setPApplicationName("libepseon_gpu")
                    .setEngineVersion(version)
                    .setPEngineName("libepseon_gpu");

                std::vector<const char*> instanceExtensions{
                    VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
                };

                auto instanceCreateInfo =
                    vk::InstanceCreateInfo()
                        .setFlags({vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR}
                        )
                        .setEnabledExtensionCount(instanceExtensions.size())
                        .setPEnabledExtensionNames(instanceExtensions)
                        .setPApplicationInfo(applicationInfo.get());
                auto instance = std::make_unique<vk::raii::Instance>(
                    std::move(context->createInstance(instanceCreateInfo))
                );

                return std::unique_ptr<VulkanApplication>(new VulkanApplication{
                    logger,
                    std::move(context),
                    std::move(applicationInfo),
                    std::move(instance)
                });
            }

            std::string VulkanApplication::getVulkanAPIVersion() {
                auto apiVersion = this->context->enumerateInstanceVersion();
                return common::vulkan_version_to_string(apiVersion);
            }

            std::vector<PhysicalDeviceInfo>
            VulkanApplication::getPhysicalDevicesInfo() {
                std::vector<PhysicalDeviceInfo> devices;

                for (auto deviceInfo : this->instance->enumeratePhysicalDevices()) {
                    auto deviceProperties = deviceInfo.getProperties();
                    auto memoryProperties = deviceInfo.getMemoryProperties();
                    devices.push_back({deviceProperties, memoryProperties});
                }

                return devices;
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
