#include "epseon_gpu/libgpu.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_core.h"
#include "vulkan/vulkan_handles.hpp"
#include "vulkan/vulkan_raii.hpp"
#include "vulkan/vulkan_structs.hpp"
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

SHARED_EXPORT void hello() {
    std::cout << "hello world!" << std::endl;
}

PyMODINIT_FUNC PyInit__libepseon_gpu(void) {
    return PyModule_Create(&libepseon_gpu);
}
namespace epseon {
    namespace gpu {

        VulkanApplication::VulkanApplication(
            std::shared_ptr<spdlog::logger>      logger,
            std::unique_ptr<vk::raii::Context>   context,
            std::unique_ptr<vk::ApplicationInfo> application_info,
            std::unique_ptr<vk::raii::Instance>  instance
        )
            : logger(logger),
              context(std::move(context)),
              application_info(std::move(application_info)),
              instance(std::move(instance)) {}

        std::optional<std::unique_ptr<VulkanApplication>>
        VulkanApplication::create(uint32_t version) {
            auto logger =
                spdlog::basic_logger_mt("libepseon_gpu", "./log/libepseon_gpu/log.txt");

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

            auto instanceCreateInfo = vk::InstanceCreateInfo()
                                          .setFlags({})
                                          .setEnabledExtensionCount(0)
                                          .setPEnabledExtensionNames({})
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
            auto apiVersion = context->enumerateInstanceVersion();

            std::stringstream ss;
            ss << vk::apiVersionVariant(apiVersion) << "."
               << vk::apiVersionMajor(apiVersion) << "."
               << vk::apiVersionMinor(apiVersion) << "."
               << vk::apiVersionPatch(apiVersion);

            return ss.str();
        }

    } // namespace gpu
} // namespace epseon
