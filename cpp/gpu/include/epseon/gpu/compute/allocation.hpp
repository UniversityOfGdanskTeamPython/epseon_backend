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

        [[nodiscard]] static constexpr vk::BufferUsageFlags getBufferUsageFlags() {
            return static_cast<vk::BufferUsageFlags>(bufferUsageFlags);
        }

        [[nodiscard]] static constexpr vma::AllocationCreateFlags getAllocationFlags() {
            return static_cast<vma::AllocationCreateFlags>(allocationFlags);
        }

        [[nodiscard]] vk::DescriptorType getDescriptorType() const {
            return vk::DescriptorType::eStorageBuffer;
        }

        [[nodiscard]] uint32_t getAllocationTotalSizeBytes(const layoutT& layout) const {
            return getScaling().getAllocationTotalSizeBytes(layout.getTotalSizeBytes());
        }

        [[nodiscard]] uint32_t getAllocationBufferCount(const layoutT& /*layout*/) const {
            return getScaling().getAllocationBufferCount();
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

        [[nodiscard]] std::vector<vk::Buffer>& getBuffers() {
            return this->buffers;
        }

        [[nodiscard]] std::vector<vk::Buffer>& getBuffers() const {
            return this->buffers;
        }

        void bind(std::shared_ptr<environment::Device>& devicePointer,
                  std::shared_ptr<scaling::Base>&       scalingPointer) {
            this->devicePointer  = devicePointer;
            this->scalingPointer = scalingPointer;
        }

        void allocateBuffers(const layoutT& layout) noexcept {
            auto bufferSizeBytes = getAllocationTotalSizeBytes(layout);
            auto bufferCreateInfo =
                vk::BufferCreateInfo().setUsage(getBufferUsageFlags()).setSize(bufferSizeBytes);
            auto allocationCreateInfo = vma::AllocationCreateInfo()
                                            .setUsage(vma::MemoryUsage::eAuto)
                                            .setFlags(getAllocationFlags());

            auto bufferCount = getAllocationBufferCount();

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

        void deallocateBuffers(const layoutT& /*layout*/) {
            auto bufferCount = getScaling().getAllocationBufferCount();

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
            auto bufferCount = getAllocationBufferCount();
            return vk::DescriptorSetLayoutBinding()
                .setBinding(layout.getBinding())
                .setDescriptorCount(bufferCount)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        }

        [[nodiscard]] vk::DescriptorPoolSize
        getDescriptorPoolSize(const layoutT& /*layout*/) const {
            return vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(getAllocationBufferCount());
        }

        [[nodiscard]] std::vector<vk::DescriptorBufferInfo>
        getBufferInfo(const layoutT& layout) const {
            uint32_t bufferCount     = getScaling().getAllocationBufferCount(layout);
            uint32_t bufferTotalSize = getScaling().getAllocationTotalSizeBytes(layout);

            std::vector<vk::DescriptorBufferInfo> bufferInfo{};
            bufferInfo.reserve(bufferCount);

            for (const vk::Buffer& buffer : buffers) {
                bufferInfo.emplace_back(buffer, 0, bufferTotalSize);
            }
        }

        template <allocation::Concept destinationBufferT>
        void recordCopyBuffer(destinationBufferT&      destination,
                              vk::raii::CommandBuffer& commandBuffer,
                              layoutT&                 layout) {
            std::vector<vk::Buffer>& sourceBuffers      = this->getBuffers();
            std::vector<vk::Buffer>& destinationBuffers = destination.getBuffers();

            uint32_t bufferCount     = getScaling().getAllocationBufferCount(layout);
            uint32_t bufferTotalSize = getScaling().getAllocationTotalSizeBytes(layout);

            auto bufferCopyInfo =
                vk::BufferCopy().setSize(bufferTotalSize).setSrcOffset(0).setDstOffset(0);

            for (uint32_t bufferIndex = 0; bufferIndex < bufferCount; bufferIndex++) {
                commandBuffer.copyBuffer(
                    sourceBuffers[bufferIndex], destinationBuffers[bufferIndex], {bufferCopyInfo});
            }
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