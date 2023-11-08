
#include "epseon_gpu/vulkan_application.hpp"
#include "spdlog/sinks/basic_file_sink.h"

namespace epseon {
    namespace gpu {
        namespace cpp {
            PhysicalDeviceInfo::PhysicalDeviceInfo(
                vk::PhysicalDeviceProperties deviceProperties
            ) :
                deviceProperties(deviceProperties) {}

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

            std::optional<std::unique_ptr<VulkanApplication>>
            VulkanApplication::create(uint32_t version) {
                auto logger = spdlog::basic_logger_mt(
                    "libepseon_gpu", "./log/libepseon_gpu/log.txt"
                );

                auto context = std::make_unique<vk::raii::Context>();

                uint32_t apiVersion = context->enumerateInstanceVersion();
                if (apiVersion < VK_API_VERSION_1_3) {
                    logger->warn(
                        "Unsupported Vulkan version: {}.{}.{}.{}",
                        vk::apiVersionVariant(apiVersion),
                        vk::apiVersionMajor(apiVersion),
                        vk::apiVersionMinor(apiVersion),
                        vk::apiVersionPatch(apiVersion)
                    );
                } else {
                    logger->info(
                        "Discovered Vulkan version: {}.{}.{}.{}",
                        vk::apiVersionVariant(apiVersion),
                        vk::apiVersionMajor(apiVersion),
                        vk::apiVersionMinor(apiVersion),
                        vk::apiVersionPatch(apiVersion)
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

                return {std::unique_ptr<VulkanApplication>(new VulkanApplication{
                    logger,
                    std::move(context),
                    std::move(applicationInfo),
                    std::move(instance)
                })};
            }

            std::string VulkanApplication::getVulkanAPIVersion() {
                auto apiVersion = this->context->enumerateInstanceVersion();

                std::stringstream ss;
                ss << vk::apiVersionVariant(apiVersion) << "."
                   << vk::apiVersionMajor(apiVersion) << "."
                   << vk::apiVersionMinor(apiVersion) << "."
                   << vk::apiVersionPatch(apiVersion);

                return ss.str();
            }

            std::vector<PhysicalDeviceInfo>
            VulkanApplication::getPhysicalDevicesInfo() {
                std::vector<PhysicalDeviceInfo> devices;

                for (auto deviceInfo : this->instance->enumeratePhysicalDevices()) {
                    auto deviceProperties = deviceInfo.getProperties();

                    devices.push_back({deviceProperties});
                }

                return {};
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
