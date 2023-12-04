#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/task_configurator/algorithm_config.hpp"

#include "epseon/gpu/algorithms/algorithm.hpp"
#include "epseon/gpu/task_handle.hpp"
#include "spdlog/fmt/bundled/core.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unistd.h>
#include <utility>
#include <vector>

namespace epseon::gpu::cpp {

    class Interrupted : public std::exception {};

    template <typename FP>
    class VibwaAlgorithm : public Algorithm<FP> {
        static_assert(std::is_floating_point<FP>::value, "FP must be an floating-point type.");

      public: /* Public constructors. */
        VibwaAlgorithm() :
            Algorithm<FP>() {}

      public: /* Public methods. */
        struct ShaderResources {
            std::vector<vk::Buffer>          stagingBuffers                 = {};
            std::vector<vma::Allocation>     stagingBuffersAllocations      = {};
            std::vector<vma::AllocationInfo> stagingBuffersAllocationsInfos = {};

            std::vector<vk::Buffer>      gpuOnlyStorageBuffers            = {};
            std::vector<vma::Allocation> gpuOnlyStorageBuffersAllocations = {};

            std::vector<vk::Buffer>          outputBuffers                 = {};
            std::vector<vma::Allocation>     outputBuffersAllocations      = {};
            std::vector<vma::AllocationInfo> outputBuffersAllocationsInfos = {};

            // Copy constructor
            ShaderResources() = default;

            // Copy constructor
            ShaderResources(const ShaderResources&) = delete;

            // Copy assignment operator
            ShaderResources& operator=(const ShaderResources&) = delete;

            ShaderResources(ShaderResources&& other) noexcept :
                stagingBuffers(std::move(other.stagingBuffers)),
                stagingBuffersAllocations(std::move(other.stagingBuffersAllocations)),
                stagingBuffersAllocationsInfos(std::move(other.stagingBuffersAllocationsInfos)),

                gpuOnlyStorageBuffers(std::move(other.gpuOnlyStorageBuffers)),
                gpuOnlyStorageBuffersAllocations(std::move(other.gpuOnlyStorageBuffersAllocations)),
                outputBuffers(std::move(other.outputBuffers)),

                outputBuffersAllocations(std::move(other.outputBuffersAllocations)),
                outputBuffersAllocationsInfos(std::move(other.outputBuffersAllocationsInfos)) {}

            // Move assignment operator
            ShaderResources& operator=(ShaderResources&& other) noexcept {
                if (this != &other) {
                    // Move resources
                    stagingBuffers            = std::move(other.stagingBuffers);
                    stagingBuffersAllocations = std::move(other.stagingBuffersAllocations);
                    stagingBuffersAllocationsInfos =
                        std::move(other.stagingBuffersAllocationsInfos);

                    gpuOnlyStorageBuffers = std::move(other.gpuOnlyStorageBuffers);
                    gpuOnlyStorageBuffersAllocations =
                        std::move(other.gpuOnlyStorageBuffersAllocations);

                    outputBuffers                 = std::move(other.outputBuffers);
                    outputBuffersAllocations      = std::move(other.outputBuffersAllocations);
                    outputBuffersAllocationsInfos = std::move(other.outputBuffersAllocationsInfos);
                }
                return *this;
            }

            ~ShaderResources() = default;

            static ShaderResources create(
                const vma::Allocator& allocator, const ShaderBuffersRequirements<FP>& requirements
            ) {
                ShaderResources resources{};

                resources.stagingBuffers.resize(requirements.stagingBuffersCount);
                resources.stagingBuffersAllocations.resize(requirements.stagingBuffersCount);
                resources.stagingBuffersAllocationsInfos.resize(requirements.stagingBuffersCount);

                resources.gpuOnlyStorageBuffers.resize(requirements.gpuOnlyStorageBuffersCount);
                resources.gpuOnlyStorageBuffersAllocations.resize(
                    requirements.gpuOnlyStorageBuffersCount
                );

                resources.outputBuffers.resize(requirements.outputBuffersCount);
                resources.outputBuffersAllocations.resize(requirements.outputBuffersCount);
                resources.outputBuffersAllocationsInfos.resize(requirements.outputBuffersCount);

                /* Allocate staging buffers. */
                for (uint32_t i = 0; i < requirements.stagingBuffersCount; i++) {
                    vma::AllocationInfo info{};

                    auto [buffer, allocation] = allocator.createBuffer(
                        vk::BufferCreateInfo()
                            .setSize(requirements.getStagingBuffersSizeBytes())
                            .setUsage(vk::BufferUsageFlagBits::eTransferSrc),
                        vma::AllocationCreateInfo()
                            .setUsage(vma::MemoryUsage::eAuto)
                            .setFlags(
                                vma::AllocationCreateFlagBits::eHostAccessSequentialWrite |
                                vma::AllocationCreateFlagBits::eMapped
                            ),
                        info
                    );

                    resources.stagingBuffers.push_back(buffer);
                    resources.stagingBuffersAllocations.push_back(allocation);
                    resources.stagingBuffersAllocationsInfos.push_back(info);
                }
                /* Allocate GPU only buffers. */
                for (uint32_t i = 0; i < requirements.gpuOnlyStorageBuffersCount; i++) {
                    auto [buffer, allocation] = allocator.createBuffer(
                        vk::BufferCreateInfo()
                            .setSize(requirements.getGpuOnlyStorageBufferSizeBytes())
                            .setUsage(
                                vk::BufferUsageFlagBits::eTransferDst |
                                vk::BufferUsageFlagBits::eStorageBuffer
                            ),
                        vma::AllocationCreateInfo()
                            .setUsage(vma::MemoryUsage::eAuto)
                            .setFlags(vma::AllocationCreateFlagBits::eDedicatedMemory)
                    );

                    resources.gpuOnlyStorageBuffers.push_back(buffer);
                    resources.gpuOnlyStorageBuffersAllocations.push_back(allocation);
                }
                /* Allocate output (read back) buffers. */
                for (uint32_t i = 0; i < requirements.outputBuffersCount; i++) {
                    vma::AllocationInfo info{};

                    auto [buffer, allocation] = allocator.createBuffer(
                        vk::BufferCreateInfo()
                            .setSize(requirements.getOutputBufferSizeBytes())
                            .setUsage(vk::BufferUsageFlagBits::eStorageBuffer),
                        vma::AllocationCreateInfo()
                            .setUsage(vma::MemoryUsage::eAuto)
                            .setFlags(
                                vma::AllocationCreateFlagBits::eHostAccessRandom |
                                vma::AllocationCreateFlagBits::eMapped
                            ),
                        info
                    );

                    resources.outputBuffers.push_back(buffer);
                    resources.outputBuffersAllocations.push_back(allocation);
                    resources.outputBuffersAllocationsInfos.push_back(info);
                }

                return resources;
            }

            void destroy(const vma::Allocator& allocator) {
                destroy(allocator, stagingBuffers, stagingBuffersAllocations);
                destroy(allocator, gpuOnlyStorageBuffers, gpuOnlyStorageBuffersAllocations);
                destroy(allocator, outputBuffers, outputBuffersAllocations);
            }

            template <typename BufferT, typename AllocationT>
            void destroy(
                const vma::Allocator&    allocator,
                std::vector<BufferT>     buffers,
                std::vector<AllocationT> allocations
            ) {
                assert(buffers.size() == allocations.size());

                for (uint32_t i = 0; i < buffers.size(); i++) {
                    allocator.destroyBuffer(buffers[i], allocations[i]);
                }
                buffers.clear();
                allocations.clear();
            }
        };

        struct ComputeBatchResources {
            vma::Allocator               allocator       = {};
            std::vector<ShaderResources> shaderResources = {};

            // Copy constructor
            ComputeBatchResources() = default;

            // Copy constructor
            explicit ComputeBatchResources(vma::Allocator allocator) :
                allocator(allocator){};

            // Copy constructor
            ComputeBatchResources(const ComputeBatchResources&) = delete;

            // Copy assignment operator
            ComputeBatchResources& operator=(const ComputeBatchResources&) = delete;

            ComputeBatchResources(ComputeBatchResources&& other) noexcept :
                shaderResources(std::move(other.shaderResources)) {}

            // Move assignment operator
            ComputeBatchResources& operator=(ComputeBatchResources&& other) noexcept {
                if (this != &other) {
                    // Move resources
                    shaderResources = std::move(other.shaderResources);
                }
                return *this;
            }

            ~ComputeBatchResources() {
                destroyResources();
                allocator.destroy();
            };

            static ComputeBatchResources create(
                uint32_t                  vulkanApiVersion,
                const vk::Instance&       instance,
                const vk::PhysicalDevice& physicalDevice,
                const vk::Device&         logicalDevice
            ) {
                ComputeBatchResources resources{
                    vma::createAllocator(vma::AllocatorCreateInfo()
                                             .setVulkanApiVersion(vulkanApiVersion)
                                             .setInstance(instance)
                                             .setPhysicalDevice(physicalDevice)
                                             .setDevice(logicalDevice))
                };
                return resources;
            }

            void allocateResources(const std::vector<ShaderBuffersRequirements<FP>>& requirements) {
                shaderResources.reserve(requirements.size());

                for (const auto& requirementStruct : requirements) {
                    shaderResources.push_back(ShaderResources::create(allocator, requirementStruct)
                    );
                }
            }

            void destroyResources() {
                for (uint32_t i = 0; i < shaderResources.size(); i++) {
                    shaderResources[i].destroy(allocator);
                }
                shaderResources.clear();
            }
        };

        virtual void run(const std::stop_token& stop_token, TaskHandle<FP>* handle) {

            const auto& physicalDevice = handle->getDeviceInterface().getPhysicalDevice();
            auto        logical_device = createLogicalDevice(physicalDevice);
            const auto& compute_context_state =
                handle->getDeviceInterface().getComputeContextState();

            auto configurator = handle->getTaskConfigurator();

            ComputeBatchResources resources = ComputeBatchResources::create(
                handle->getDeviceInterface().getComputeContextState().getVulkanApiVersion(),
                *compute_context_state.getVkInstance(),
                *physicalDevice,
                *logical_device
            );
            resources.allocateResources(configurator.getShaderBufferRequirements());

            /*
            size_t allocation_block_size = handle->getTaskConfigurator()
                                               .getHardwareConfig()
                                               ->getAllocationBlockSize();

            // auto buffer = logical_device.createBuffer(
            //     vk::BufferCreateInfo()
            //         .setSize(buffer_size_bytes)
            //         .setUsage(vk::BufferUsageFlagBits::eStorageBuffer)
            // );
            // buffer.bindMemory();
            // auto requirements =
            //     logical_device.getBufferMemoryRequirements(buffer);
            //  requirements.

            auto descriptor_sets =
                allocateDescriptorSets(logical_device, group_size, 0);

            std::vector<vk::DescriptorBufferInfo> buffer_info{group_size};
            std::vector<vk::WriteDescriptorSet>   write_descriptor_sets{
                group_size};

            for (uint32_t i = 0; i < group_size; i++)
            {
                buffer_info[i].setOffset(0).setRange(VK_WHOLE_SIZE);
                write_descriptor_sets[i]
                    .setDstSet(*descriptor_sets[0])
                    .setDstBinding(0)
                    .setDstArrayElement(i)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setDescriptorCount(1)
                    .setBufferInfo(buffer_info);
            } */

            std::cout << "Thread finished" << std::endl;
        }

        uint32_t selectQueueFamilyIndex(const vk::raii::PhysicalDevice& physicalDevice) {
            uint32_t queue_family_index = -1;

            for (auto i = 0; const auto& queue : physicalDevice.getQueueFamilyProperties()) {
                if ((queue.queueFlags & vk::QueueFlagBits::eCompute) &&
                    (queue.queueFlags & vk::QueueFlagBits::eTransfer)) {
                    queue_family_index = i;
                    break;
                }
                i++;
            }

            assert(queue_family_index != -1);
            return queue_family_index;
        }

        vk::raii::Device createLogicalDevice(const vk::raii::PhysicalDevice& physicalDevice) {
            // Exclusive transfer Queue in some GPUs, we may use it in
            // future.
            uint32_t             queue_index      = selectQueueFamilyIndex(physicalDevice);
            std::array<float, 1> queue_priorities = {1.0F};
            std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{
                vk::DeviceQueueCreateInfo()
                    .setQueueFamilyIndex(queue_index)
                    .setQueueCount(1)
                    .setPQueuePriorities(queue_priorities.data())
            };

            std::vector<const char*> required_device_extensions{};

            auto logicalDevice = physicalDevice.createDevice(
                vk::DeviceCreateInfo()
                    .setQueueCreateInfos(queueCreateInfos)
                    .setPEnabledExtensionNames(required_device_extensions)
            );
            return std::move(logicalDevice);
        }

        std::vector<vk::raii::DescriptorSet> allocateDescriptorSets(
            vk::raii::Device& logical_device, uint32_t descriptorCount, uint32_t binding
        ) {

            constexpr uint32_t descriptor_set_count = 0;

            std::array<vk::DescriptorSetLayoutBinding, 1> descriptorSetLayoutBindings{
                vk::DescriptorSetLayoutBinding()
                    .setBinding(0)
                    .setDescriptorCount(descriptorCount)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setStageFlags({vk::ShaderStageFlagBits::eCompute})
            };

            auto descriptorSetLayout = logical_device.createDescriptorSetLayout(
                vk::DescriptorSetLayoutCreateInfo().setBindings(descriptorSetLayoutBindings)
            );

            std::array<vk::DescriptorPoolSize, 1> descriptorPoolSizes{
                vk::DescriptorPoolSize()
                    .setDescriptorCount(descriptorCount)
                    .setType(vk::DescriptorType::eStorageBuffer)
            };

            auto descriptorPool =
                logical_device.createDescriptorPool(vk::DescriptorPoolCreateInfo()
                                                        .setPoolSizes(descriptorPoolSizes)
                                                        .setMaxSets(descriptor_set_count));

            return logical_device.allocateDescriptorSets(
                vk::DescriptorSetAllocateInfo()
                    .setDescriptorPool(*descriptorPool)
                    .setSetLayouts(*descriptorSetLayout)
                    .setDescriptorSetCount(descriptor_set_count)
            );
        }
    };
} // namespace epseon::gpu::cpp
