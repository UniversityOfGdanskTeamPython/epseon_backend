#pragma once

#include "epseon/vulkan_headers.hpp"

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
#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace epseon::gpu::cpp {

    namespace environment {
        class Device;
    }

    namespace shader {

        template <resources::Concept resourceT>
        class Base {
          public:
            Base() = delete;

            Base(std::shared_ptr<environment::Device>& devicePtr_,
                 std::shared_ptr<scaling::Base>&       scalingPtr_) :
                devicePtr(devicePtr_),
                scalingPtr(scalingPtr_){};

            Base(const Base&)     = delete;
            Base(Base&&) noexcept = default;

            virtual ~Base() = default;

            Base& operator=(const Base&)     = delete;
            Base& operator=(Base&&) noexcept = default;

            virtual void run() = 0;

          private:
            std::shared_ptr<environment::Device> devicePtr  = {};
            std::shared_ptr<scaling::Base>       scalingPtr = {};
        };

        class Dynamic : public Base<resources::Dynamic> {
          public:
            Dynamic() = delete;

            Dynamic(resources::Dynamic&&                  resource_,
                    std::shared_ptr<environment::Device>& devicePtr_,
                    std::shared_ptr<scaling::Base>&       scalingPtr_) :
                Base<resources::Dynamic>(devicePtr_, scalingPtr_),
                resource(std::move(resource_)){};

            Dynamic(const Dynamic&)     = delete;
            Dynamic(Dynamic&&) noexcept = default;

            ~Dynamic() override = default;

            Dynamic& operator=(const Dynamic&)     = delete;
            Dynamic& operator=(Dynamic&&) noexcept = default;

            void run() override {
                resource.prepare(this->devicePtr, this->scalingPtr);
            }

          private:
            resources::Dynamic resource;
        };

        template <typename resourceT>
        class Static : public Base<resourceT> {
          public:
            Static() = delete;

            Static(resourceT&&                           resource_,
                   std::shared_ptr<environment::Device>& devicePtr_,
                   std::shared_ptr<scaling::Base>&       scalingPtr_) :
                Base<resourceT>(devicePtr_, scalingPtr_),
                resource(resource_){};

            Static(const Static&)     = delete;
            Static(Static&&) noexcept = default;

            ~Static() override = default;

            Static& operator=(const Static&)     = delete;
            Static& operator=(Static&&) noexcept = default;

            void run() override {
                resource.prepare(this->devicePtr, this->scalingPtr);
            }

          private:
            resourceT resource;
        };
    } // namespace shader

    namespace environment {
        class Device {
          public:
            Device(const Device&)     = delete;
            Device(Device&&) noexcept = default;

            ~Device() {
                // Vulkan Memory Allocator requires manual cleanup invocation.
                allocator->destroy();
            }

            Device& operator=(const Device&)     = delete;
            Device& operator=(Device&&) noexcept = default;

            static std::shared_ptr<Device> create(uint32_t deviceId) {
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

                return std::make_shared<Device>(
                    deviceInfo.shaderStorageBufferArrayNonUniformIndexing,
                    vulkanApiVersion,
                    std::move(context),
                    std::move(instance),
                    std::move(physicalDevice),
                    std::move(deviceInfo.logicalDevice),
                    std::make_shared<vma::Allocator>(allocator));
            }

          private:
            static vk::raii::PhysicalDevice createPhysicalDevice(vk::raii::Instance& instance,
                                                                 uint32_t            deviceId) {
                for (vk::raii::PhysicalDevice& physicalDevice :
                     instance.enumeratePhysicalDevices()) {
                    auto props = physicalDevice.getProperties();
                    if (props.deviceID == deviceId) {
                        return physicalDevice;
                    }
                }
                auto str = fmt::format("Device with ID {} not found", deviceId);
                throw std::runtime_error(str);
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
                shaderStorageBufferArrayNonUniformIndexing = static_cast<bool>(
                    deviceFeatures.features.shaderStorageBufferArrayDynamicIndexing);

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

          private:
            Device(bool                       supportsBufferArrayScaling_,
                   uint32_t                   apiVersion_,
                   vk::raii::Context&&        context_,
                   vk::raii::Instance&&       instance_,
                   vk::raii::PhysicalDevice&& physicalDevice_,
                   vk::raii::Device&&         logicalDevice_,
                   uint32_t                   queueIndex_,
                   vma::Allocator&&           allocator_) :
                supportsBufferArrayScaling(supportsBufferArrayScaling_),
                apiVersion(apiVersion_),
                context(std::move(context_)),
                instance(std::move(instance_)),
                physicalDevice(std::move(physicalDevice_)),
                logicalDevice(std::move(logicalDevice_)),
                queueIndex(queueIndex_),
                allocator(std::make_shared<vma::Allocator>(allocator_)) {}

          public:
            [[nodiscard]] std::shared_ptr<scaling::Base> getOptimalScalingPolicy() const {
                if (supportsBufferArrayScaling) {
                    return std::make_shared<scaling::BufferArray>();
                } else {
                    return std::make_shared<scaling::LargeBuffer>();
                }
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
    } // namespace environment

    class VibwaResources : public resources::Static {
      public:
        explicit VibwaResources(uint64_t batchSize_) :
            configuration({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 0}}),
            y({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 1}}),
            buffer0({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 2}}),
            output({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 3}}) {}

        template <typename CallableT>
        void iterateBuffers(CallableT callable) {
            callable(configuration);
            callable(y);
            callable(buffer0);
            callable(output);
        }

      private:
        struct Configuration {
            float    mass_atom_0               = {};
            float    mass_atom_1               = {};
            float    integration_step          = {};
            float    min_distance_to_asymptote = {};
            uint32_t min_level                 = {};
            uint32_t max_level                 = {};
        };

        buffer::HostToDevice<layout::Static<Configuration>> configuration;
        buffer::HostToDevice<layout::Static<float>>         y;
        buffer::DeviceLocal<layout::Static<float>>          buffer0;
        buffer::DeviceToHost<layout::Static<float>>         output;
    };

    template <typename T>
    void foo() { // NOLINT(misc-definitions-in-headers)
        const uint64_t batchSize  = 128;
        const uint64_t bufferSize = 1024;

        std::shared_ptr<environment::Device> device = environment::Device::create(0);

        std::shared_ptr<scaling::Base> scalingPolicy = device->getOptimalScalingPolicy();

        shader::Dynamic shader1{resources::Dynamic{{layout::Dynamic{{.batchSize = batchSize,
                                                                     .itemCount = bufferSize,
                                                                     .itemSize  = sizeof(float),
                                                                     .set       = 0,
                                                                     .binding   = 0}}},
                                                   {},
                                                   {}},
                                device,
                                scalingPolicy};

        shader1.run();

        shader::Static<VibwaResources> shader2{VibwaResources{batchSize}, device, scalingPolicy};
    }
} // namespace epseon::gpu::cpp
