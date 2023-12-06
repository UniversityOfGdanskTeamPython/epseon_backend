#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/predecl.hpp"

#include "spdlog/logger.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace epseon::gpu::cpp {

    struct ComputeContextState {
      public: /* Public members. */
        std::shared_ptr<spdlog::logger>      logger           = {};
        std::shared_ptr<vk::raii::Context>   context          = {};
        std::shared_ptr<vk::ApplicationInfo> application_info = {};
        std::shared_ptr<vk::raii::Instance>  instance         = {};

      public: /* Public constructors. */
        ComputeContextState(ComputeContextState&);
        ComputeContextState(std::shared_ptr<spdlog::logger>, std::shared_ptr<vk::raii::Context>, std::shared_ptr<vk::ApplicationInfo>, std::shared_ptr<vk::raii::Instance>);

      public: /* Public destructor. */
        ~ComputeContextState() = default;

      public: /* Public methods. */
        [[nodiscard]] const vk::raii::Instance& getVkInstance() const {
            return *this->instance;
        }

        [[nodiscard]] uint32_t getVulkanApiVersion() const {
            return this->application_info->apiVersion;
        }
    };

    struct PhysicalDeviceInfo {
      public: /* Public members. */
        vk::PhysicalDeviceProperties       deviceProperties;
        vk::PhysicalDeviceMemoryProperties memoryProperties;
    };

    class ComputeDeviceInterface;

    class ComputeContext {
      private:
        std::shared_ptr<ComputeContextState> state;

      public: /* Public constructors. */
        ComputeContext(
            std::shared_ptr<spdlog::logger>      logger_,
            std::shared_ptr<vk::raii::Context>   context_,
            std::shared_ptr<vk::ApplicationInfo> application_info_,
            std::shared_ptr<vk::raii::Instance>  instance_
        );

      public: /* Public destructor. */
        virtual ~ComputeContext() = default;

      public: /* Public factory method. */
        static std::shared_ptr<ComputeContext>
        create(uint32_t version = VK_MAKE_API_VERSION(0, 0, 1, 0));

      public: /* Public methods. */
        std::string                             getVulkanAPIVersion();
        std::vector<PhysicalDeviceInfo>         getPhysicalDevicesInfo();
        std::shared_ptr<ComputeDeviceInterface> getDeviceInterface(uint32_t);
    };
} // namespace epseon::gpu::cpp
