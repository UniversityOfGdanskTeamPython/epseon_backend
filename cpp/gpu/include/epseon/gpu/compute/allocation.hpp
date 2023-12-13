#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

#include "epseon/gpu/compute/layout.hpp"
#include "epseon/gpu/compute/scaling.hpp"
#include "vk_mem_alloc.h"
#include "vk_mem_alloc_handles.hpp"
#include "vk_mem_alloc_structs.hpp"
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace epseon::gpu::cpp::environment {
    class Device;
}

namespace epseon::gpu::cpp::allocation {
    template <VkBufferUsageFlags       bufferUsageFlags,
              VmaAllocationCreateFlags allocationFlags,
              layout::Concept          layoutT>
    struct Allocation {

        using layout_type = layoutT;

        [[nodiscard]] scaling::Base& getScaling() {
            assert(scalingPointer);
            return *this->scalingPointer;
        }

        [[nodiscard]] const scaling::Base& getScaling() const {
            assert(scalingPointer);
            return *this->scalingPointer;
        }

        [[nodiscard]] environment::Device& getDevice() {
            assert(devicePointer);
            return *this->devicePointer;
        }

        [[nodiscard]] const environment::Device& getDevice() const {
            assert(devicePointer);
            return *this->devicePointer;
        }

        [[nodiscard]] static constexpr vk::BufferUsageFlags getBufferUsageFlags() {
            return static_cast<vk::BufferUsageFlags>(bufferUsageFlags);
        }

        [[nodiscard]] static constexpr vma::AllocationCreateFlags getAllocationFlags() {
            return static_cast<vma::AllocationCreateFlags>(allocationFlags);
        }

        void bind(std::shared_ptr<environment::Device>& devicePointer,
                  std::shared_ptr<scaling::Base>&       scalingPointer) {
            this->devicePointer  = devicePointer;
            this->scalingPointer = scalingPointer;
        }

        void allocateBuffers(const layoutT& layout) noexcept {
            auto bufferSizeBytes = getScaling().getAllocationTotalSizeBytes(
                layout.getTotalSizeBytes(), layout.getBatchSize());
            auto bufferCreateInfo =
                vk::BufferCreateInfo().setUsage(getBufferUsageFlags()).setSize(bufferSizeBytes);
            auto allocationCreateInfo = vma::AllocationCreateInfo()
                                            .setUsage(vma::MemoryUsage::eAuto)
                                            .setFlags(getAllocationFlags());

            auto bufferCount = getScaling().getAllocationBufferCount(layout.getBatchSize());

            buffers.reserve(bufferCount);
            allocations.reserve(bufferCount);
            allocationInfos.reserve(bufferCount);

            for (uint64_t i = 0; i < bufferCount; i++) {
                vma::AllocationInfo info{};

                auto [buffer, allocation] = getDevice().getDeviceAllocator().createBuffer(
                    bufferCreateInfo, allocationCreateInfo, &info);

                buffers.push_back(std::move(buffer));
                allocations.push_back(std::move(allocation));
                allocationInfos.push_back(info);
            }
        }

        void deallocateBuffers(const layoutT& layout) {
            auto bufferCount = getScaling().getAllocationBufferCount(layout.getBatchSize());

            for (uint32_t i = 0; i < bufferCount; i++) {
                getDevice().getDeviceAllocator().destroyBuffer(buffers[i], allocations[i]);
            }
            buffers.clear();
            allocations.clear();
            allocationInfos.clear();
        }

        [[nodiscard]] std::optional<vk::DescriptorSetLayoutBinding>
        getDescriptorSetLayoutBindings(uint32_t setIndex, const layoutT& layout) const {
            if (setIndex != layout.getSet()) {
                return std::nullopt;
            }
            auto bufferCount = getScaling().getAllocationBufferCount(layout.getBatchSize());
            return vk::DescriptorSetLayoutBinding()
                .setBinding(layout.getBinding())
                .setDescriptorCount(bufferCount)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        }

        [[nodiscard]] vk::DescriptorPoolSize getDescriptorPoolSize(const layoutT& layout) const {
            return vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(getScaling().getAllocationBufferCount(layout.getBatchSize()));
        }

      private:
        std::vector<vk::Buffer>          buffers         = {};
        std::vector<vma::Allocation>     allocations     = {};
        std::vector<vma::AllocationInfo> allocationInfos = {};

        std::shared_ptr<environment::Device> devicePointer  = {};
        std::shared_ptr<scaling::Base>       scalingPointer = {};
    };

    template <layout::Concept layoutT>
    using HostTransferSrc =
        Allocation<VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                   layoutT>;

    template <layout::Concept layoutT>
    using DeviceTransferDst =
        Allocation<VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                   layoutT>;

    template <layout::Concept layoutT>
    using DeviceLocal = Allocation<VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                   layoutT>;

    template <layout::Concept layoutT>
    using DeviceTransferSrc =
        Allocation<VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                   layoutT>;

    template <layout::Concept layoutT>
    using HostTransferDst =
        Allocation<VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                   layoutT>;
} // namespace epseon::gpu::cpp::allocation