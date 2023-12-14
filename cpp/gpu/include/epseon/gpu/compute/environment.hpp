#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

#include "epseon/gpu/common.hpp"
#include "epseon/gpu/compute/allocation.hpp"
#include "epseon/gpu/compute/buffer.hpp"
#include "epseon/gpu/compute/layout.hpp"
#include "epseon/gpu/compute/resources.hpp"
#include "epseon/gpu/compute/scaling.hpp"

#include "fmt/format.h"
#include "vk_mem_alloc.h"
#include "vk_mem_alloc_handles.hpp"
#include "vk_mem_alloc_structs.hpp"
#include <optional>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace epseon::gpu::cpp::environment {

    class Device {
      public:
        Device(bool                       supportsBufferArrayScaling_,
               uint32_t                   apiVersion_,
               vk::raii::Context&&        context_,
               vk::raii::Instance&&       instance_,
               vk::raii::PhysicalDevice&& physicalDevice_,
               vk::raii::Device&&         logicalDevice_,
               uint32_t                   queueIndex_,
               vma::Allocator             allocator_) :
            supportsBufferArrayScaling(supportsBufferArrayScaling_),
            apiVersion(apiVersion_),
            context(std::move(context_)),
            instance(std::move(instance_)),
            physicalDevice(std::move(physicalDevice_)),
            logicalDevice(std::move(logicalDevice_)),
            queueIndex(queueIndex_),
            allocator(std::make_shared<vma::Allocator>(allocator_)) {}

        Device(const Device&)     = delete;
        Device(Device&&) noexcept = default;

        ~Device() {
            // Vulkan Memory Allocator requires manual cleanup invocation.
            allocator->destroy();
        }

        Device& operator=(const Device&)     = delete;
        Device& operator=(Device&&) noexcept = default;

        static std::shared_ptr<Device> create(std::optional<uint32_t> deviceId = std::nullopt) {
            auto     context          = vk::raii::Context();
            uint32_t vulkanApiVersion = context.enumerateInstanceVersion();

            if (vulkanApiVersion < vk::ApiVersion12) {
                auto str = fmt::format("Insufficient Vulkan version {} (Vulkan 1.2 required) "
                                       "try updating your drivers or SDK.",
                                       common::vulkan_version_to_string(vulkanApiVersion));
                throw std::runtime_error(str);
            }
            vulkanApiVersion = vk::ApiVersion12;

            auto version         = vk::makeApiVersion(0, 1, 0, 0);
            auto applicationInfo = std::make_shared<vk::ApplicationInfo>();
            applicationInfo->setApplicationVersion(version)
                .setPApplicationName("libepseon")
                .setEngineVersion(version)
                .setPEngineName("epseon_compute_core")
                .setApiVersion(vk::ApiVersion13);

            std::vector<const char*> instanceExtensions{};
            auto                     instanceCreateInfo = vk::InstanceCreateInfo()
                                          .setEnabledExtensionCount(instanceExtensions.size())
                                          .setPEnabledExtensionNames(instanceExtensions)
                                          .setPApplicationInfo(applicationInfo.get());
            auto instance = context.createInstance(instanceCreateInfo);

            vk::raii::PhysicalDevice physicalDevice = createPhysicalDevice(instance, deviceId);
            _createLogicalDevice     deviceInfo     = createLogicalDevice(physicalDevice);

            auto functions = vma::VulkanFunctions();
            functions.setVkGetInstanceProcAddr(instance.getDispatcher()->vkGetInstanceProcAddr)
                .setVkGetDeviceProcAddr(instance.getDispatcher()->vkGetDeviceProcAddr);
            vma::Allocator allocator =
                vma::createAllocator(vma::AllocatorCreateInfo()
                                         .setVulkanApiVersion(vulkanApiVersion)
                                         .setInstance(*instance)
                                         .setPhysicalDevice(*physicalDevice)
                                         .setDevice(*deviceInfo.logicalDevice)
                                         .setPVulkanFunctions(&functions));

            return std::make_shared<Device>(deviceInfo.shaderStorageBufferArrayNonUniformIndexing,
                                            vulkanApiVersion,
                                            std::move(context),
                                            std::move(instance),
                                            std::move(physicalDevice),
                                            std::move(deviceInfo.logicalDevice),
                                            deviceInfo.queueIndex,
                                            allocator);
        }

      private:
        static vk::raii::PhysicalDevice createPhysicalDevice(vk::raii::Instance&     instance,
                                                             std::optional<uint32_t> deviceId) {
            for (vk::raii::PhysicalDevice& physicalDevice : instance.enumeratePhysicalDevices()) {
                auto props = physicalDevice.getProperties();
                if (deviceId.has_value() && props.deviceID == deviceId) {
                    return physicalDevice;
                }
            }
            if (deviceId.has_value()) {
                auto str = fmt::format("Device with ID {} not found", deviceId.value());
                throw std::runtime_error(str);
            } else {
                throw std::runtime_error("No devices available.");
            }
        }

        struct _createLogicalDevice {
            vk::raii::Device logicalDevice;
            uint32_t         queueIndex;
            bool             shaderStorageBufferArrayNonUniformIndexing;
            bool             shaderStorageBufferArrayDynamicIndexing;
        };

        static _createLogicalDevice
        createLogicalDevice(const vk::raii::PhysicalDevice& physicalDevice) {
            // Exclusive transfer Queue in some GPUs, we may use it in
            // future.
            uint32_t             queueIndex      = selectQueueFamilyIndex(physicalDevice);
            std::array<float, 1> queuePriorities = {1.0F};

            bool shaderStorageBufferArrayNonUniformIndexing = false;
            bool shaderStorageBufferArrayDynamicIndexing    = false;

            std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{
                vk::DeviceQueueCreateInfo()
                    .setQueueFamilyIndex(queueIndex)
                    .setQueuePriorities(queuePriorities)};

            std::vector<const char*> requiredDeviceExtensions{};

            auto featuresChain =
                physicalDevice.getFeatures2<vk::PhysicalDeviceFeatures2,
                                            vk::PhysicalDeviceDescriptorIndexingFeatures>();

            auto deviceFeatures = featuresChain.get<vk::PhysicalDeviceFeatures2>();
            shaderStorageBufferArrayNonUniformIndexing =
                static_cast<bool>(deviceFeatures.features.shaderStorageBufferArrayDynamicIndexing);

            auto descriptorIndexingFeatures =
                featuresChain.get<vk::PhysicalDeviceDescriptorIndexingFeatures>();

            shaderStorageBufferArrayNonUniformIndexing = static_cast<bool>(
                descriptorIndexingFeatures.shaderStorageBufferArrayNonUniformIndexing);

            if (shaderStorageBufferArrayNonUniformIndexing) {
                requiredDeviceExtensions.push_back("VK_EXT_descriptor_indexing");
            }

            assert(deviceFeatures.pNext != nullptr);
            deviceFeatures.setPNext(&descriptorIndexingFeatures);

            auto logicalDevice = physicalDevice.createDevice(
                vk::DeviceCreateInfo()
                    // Enables all features which were previously declared available.
                    // This is the lazy way, proper way would be to turn on only those
                    // which you really need.
                    .setPNext(&deviceFeatures)
                    .setQueueCreateInfos(queueCreateInfos)
                    .setPEnabledExtensionNames(requiredDeviceExtensions));

            return {.logicalDevice = std::move(logicalDevice),
                    .queueIndex    = queueIndex,
                    .shaderStorageBufferArrayNonUniformIndexing =
                        shaderStorageBufferArrayNonUniformIndexing,
                    .shaderStorageBufferArrayDynamicIndexing =
                        shaderStorageBufferArrayDynamicIndexing};
        }

        static uint32_t selectQueueFamilyIndex(const vk::raii::PhysicalDevice& physicalDevice) {
            uint32_t queueFamilyIndex = -1;

            for (auto i = 0; const auto& queue : physicalDevice.getQueueFamilyProperties()) {
                if ((queue.queueFlags & vk::QueueFlagBits::eCompute) &&
                    (queue.queueFlags & vk::QueueFlagBits::eTransfer)) {
                    queueFamilyIndex = i;
                    break;
                }
                i++;
            }
            assert(queueFamilyIndex != -1);
            return queueFamilyIndex;
        }

      public:
        [[nodiscard]] std::shared_ptr<scaling::Base>
        getOptimalScalingPolicy(uint32_t batchSize) const {
            if (supportsBufferArrayScaling) {
                return std::make_shared<scaling::BufferArray>(batchSize);
            } else {
                return std::make_shared<scaling::LargeBuffer>(batchSize);
            }
        }

        [[nodiscard]] vma::Allocator& getDeviceAllocator() {
            return *this->allocator;
        }

        [[nodiscard]] const vma::Allocator& getDeviceAllocator() const {
            return *this->allocator;
        }

        [[nodiscard]] vk::raii::Device& getLogicalDevice() {
            return this->logicalDevice;
        }

        [[nodiscard]] const vk::raii::Device& getLogicalDevice() const {
            return this->logicalDevice;
        }

        [[nodiscard]] uint32_t getQueueFamilyIndex() const {
            return queueIndex;
        }

      private:
        bool                            supportsBufferArrayScaling = false;
        uint32_t                        apiVersion;
        vk::raii::Context               context;
        vk::raii::Instance              instance;
        vk::raii::PhysicalDevice        physicalDevice;
        vk::raii::Device                logicalDevice;
        uint32_t                        queueIndex;
        std::shared_ptr<vma::Allocator> allocator = {};
    };
} // namespace epseon::gpu::cpp::environment