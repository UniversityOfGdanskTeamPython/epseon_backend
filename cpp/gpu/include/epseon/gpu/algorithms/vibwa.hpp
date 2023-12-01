#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/algorithms/algorithm.hpp"
#include "epseon/gpu/task_handle.hpp"
#include "spdlog/fmt/bundled/core.h"
#include "vk_mem_alloc_enums.hpp"
#include "vk_mem_alloc_handles.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace epseon {
    namespace gpu {
        namespace cpp {

            class Interrupted : public std::exception {};

#define CHECK_AND_THROW_IF_REQUESTED_STOP         \
    if (this->stop_token_copy.stop_requested()) { \
        throw Interrupted();                      \
    }

            template <typename FP>
            class VibwaAlgorithm : public Algorithm<FP> {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              private: /* Private members. */
                std::stop_token stop_token_copy;

              public: /* Public constructors. */
                VibwaAlgorithm() :
                    Algorithm<FP>() {}

              public: /* Public destructor. */
                virtual ~VibwaAlgorithm() {}

              public: /* Public methods. */
                virtual void run(std::stop_token stop_token, TaskHandle<FP>* handle) {
                    this->stop_token_copy = stop_token;
                    CHECK_AND_THROW_IF_REQUESTED_STOP;

                    auto physical_device_ptr = handle->device->getPhysicalDevice();
                    auto logical_device      = createLogicalDevice(physical_device_ptr);
                    const auto& compute_context_state =
                        handle->getDeviceInterface().getComputeContextState();

                    vma::Allocator memory_allocator = vma::createAllocator(
                        vma::AllocatorCreateInfo()
                            .setVulkanApiVersion(handle->getDeviceInterface()
                                                     .getComputeContextState()
                                                     .application_info->apiVersion)
                            .setPhysicalDevice(*(*physical_device_ptr))
                            .setDevice(*logical_device)
                            .setInstance(*(*compute_context_state.instance))
                    );
                    size_t group_size =
                        handle->getTaskConfigurator().getHardwareConfig()->getGroupSize(
                        );

                    size_t buffer_element_count = handle->getTaskConfigurator()
                                                      .getHardwareConfig()
                                                      ->getPotentialBufferSize();
                    size_t buffer_size_bytes = buffer_element_count * sizeof(FP);

                    auto allocationCreateInfo =
                        vma::AllocationCreateInfo()
                            .setUsage(vma::MemoryUsage::eAuto)
                            .setFlags(
                                vma::AllocationCreateFlagBits::
                                    eHostAccessSequentialWrite |
                                vma::AllocationCreateFlagBits::eMapped
                            );

                    auto buffer = memory_allocator.createBuffer(
                        vk::BufferCreateInfo()
                            .setSize(buffer_size_bytes)
                            .setUsage(vk::BufferUsageFlagBits::eTransferSrc),
                        allocationCreateInfo
                    );

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

                uint32_t selectQueueFamilyIndex(
                    std::shared_ptr<vk::raii::PhysicalDevice> physical_device_ptr
                ) {
                    uint32_t queue_family_index = -1;

                    for (auto i = 0; const auto& queue :
                                     physical_device_ptr->getQueueFamilyProperties()) {
                        CHECK_AND_THROW_IF_REQUESTED_STOP;

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

                vk::raii::Device createLogicalDevice(
                    std::shared_ptr<vk::raii::PhysicalDevice> physical_device_ptr
                ) {
                    // Exclusive transfer Queue in some GPUs, we may use it in
                    // future.
                    uint32_t queue_index = selectQueueFamilyIndex(physical_device_ptr);
                    std::array<float, 1>                   queue_priorities = {1.0f};
                    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos{
                        vk::DeviceQueueCreateInfo()
                            .setQueueFamilyIndex(queue_index)
                            .setQueueCount(1)
                            .setPQueuePriorities(queue_priorities.data())
                    };

                    std::vector<const char*> required_device_extensions{};

                    auto logical_device = physical_device_ptr->createDevice(
                        vk::DeviceCreateInfo()
                            .setQueueCreateInfos(queue_create_infos)
                            .setPEnabledExtensionNames(required_device_extensions)
                    );
                    return std::move(logical_device);
                }

                std::vector<vk::raii::DescriptorSet> allocateDescriptorSets(
                    vk::raii::Device& logical_device,
                    uint32_t          descriptorCount,
                    uint32_t          binding
                ) {

                    constexpr uint32_t descriptor_set_count = 0;

                    std::array<vk::DescriptorSetLayoutBinding, 1>
                        descriptor_set_layout_bindings{
                            vk::DescriptorSetLayoutBinding()
                                .setBinding(0)
                                .setDescriptorCount(descriptorCount)
                                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                                .setStageFlags({vk::ShaderStageFlagBits::eCompute})
                        };
                    auto descriptor_set_layout =
                        logical_device.createDescriptorSetLayout(
                            vk::DescriptorSetLayoutCreateInfo().setBindings(
                                descriptor_set_layout_bindings
                            )
                        );

                    std::array<vk::DescriptorPoolSize, 1> descriptor_pool_sizes{
                        vk::DescriptorPoolSize()
                            .setDescriptorCount(descriptorCount)
                            .setType(vk::DescriptorType::eStorageBuffer)
                    };
                    auto descriptor_pool = logical_device.createDescriptorPool(
                        vk::DescriptorPoolCreateInfo()
                            .setPoolSizes(descriptor_pool_sizes)
                            .setMaxSets(descriptor_set_count)
                    );

                    auto descriptor_sets = logical_device.allocateDescriptorSets(
                        vk::DescriptorSetAllocateInfo()
                            .setDescriptorPool(*descriptor_pool)
                            .setSetLayouts(*descriptor_set_layout)
                            .setDescriptorSetCount(descriptor_set_count)
                    );
                    return std::move(descriptor_sets);
                }
            };

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
