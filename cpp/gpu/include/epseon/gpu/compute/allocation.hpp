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

        Allocation()                            = default;
        Allocation(const Allocation& other)     = delete;
        Allocation(Allocation&& other) noexcept = default;

        ~Allocation() {
            deallocateBuffers();
        }

        Allocation& operator=(const Allocation& other)     = default;
        Allocation& operator=(Allocation&& other) noexcept = default;

        [[nodiscard]] static constexpr vk::BufferUsageFlags getBufferUsageFlags() {
            return static_cast<vk::BufferUsageFlags>(bufferUsageFlags);
        }

        [[nodiscard]] static constexpr vma::AllocationCreateFlags getAllocationFlags() {
            return static_cast<vma::AllocationCreateFlags>(allocationFlags);
        }

        [[nodiscard]] vk::DescriptorType getDescriptorType() const {
            return vk::DescriptorType::eStorageBuffer;
        }

        [[nodiscard]] uint32_t getAllocationTotalSizeBytes(const layout_type& layout) const {
            return getScaling().getAllocationTotalSizeBytes(layout.getTotalSizeBytes());
        }

        [[nodiscard]] uint32_t getAllocationBufferCount() const {
            return getScaling().getAllocationBufferCount();
        }

        [[nodiscard]] uint32_t getDescriptorCount() const {
            return getAllocationBufferCount();
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

        [[nodiscard]] const std::vector<vk::Buffer>& getBuffers() const {
            return this->buffers;
        }

        void bind(std::shared_ptr<environment::Device>& devicePointer,
                  std::shared_ptr<scaling::Base>&       scalingPointer) {
            this->devicePointer  = devicePointer;
            this->scalingPointer = scalingPointer;
        }

        void allocateBuffers(const layout_type& layout) noexcept {
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

        void deallocateBuffers() {
            if (buffers.empty()) {
                return;
            }
            auto bufferCount = getAllocationBufferCount();

            for (uint32_t i = 0; i < bufferCount; i++) {
                getDevice().getDeviceAllocator().destroyBuffer(buffers[i], allocations[i]);
            }
            buffers.clear();
            allocations.clear();
            allocationInfos.clear();
        }

        [[nodiscard]] std::optional<vk::DescriptorSetLayoutBinding>
        getDescriptorSetLayoutBindings(uint32_t setIndex, const layout_type& layout) const {
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
        getDescriptorPoolSize(const layout_type& /*layout*/) const {
            return vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(getDescriptorCount());
        }

        [[nodiscard]] std::vector<vk::DescriptorBufferInfo>
        getBufferInfo(const layout_type& layout) const {
            uint32_t bufferCount     = getAllocationBufferCount();
            uint32_t bufferTotalSize = getAllocationTotalSizeBytes(layout);

            std::vector<vk::DescriptorBufferInfo> bufferInfo{};
            bufferInfo.reserve(bufferCount);

            for (const vk::Buffer& buffer : buffers) {
                bufferInfo.emplace_back(buffer, 0, bufferTotalSize);
            }
            return bufferInfo;
        }

        template <allocation::Concept destinationBufferT>
        void recordCopyBuffer(destinationBufferT&      destination,
                              vk::raii::CommandBuffer& commandBuffer,
                              layout_type&             layout) {
            std::vector<vk::Buffer>& sourceBuffers      = this->getBuffers();
            std::vector<vk::Buffer>& destinationBuffers = destination.getBuffers();

            uint32_t bufferCount     = getAllocationBufferCount();
            uint32_t bufferTotalSize = getAllocationTotalSizeBytes(layout);

            auto bufferCopyInfo =
                vk::BufferCopy().setSize(bufferTotalSize).setSrcOffset(0).setDstOffset(0);

            for (uint32_t bufferIndex = 0; bufferIndex < bufferCount; bufferIndex++) {
                commandBuffer.copyBuffer(
                    sourceBuffers[bufferIndex], destinationBuffers[bufferIndex], {bufferCopyInfo});
            }
        }

        virtual void fillBuffers(layout_type::fill_function_type fillFunction,
                                 const layout_type&              layout) {
            uint32_t bufferCount     = getAllocationBufferCount();
            uint64_t bufferTotalSize = getAllocationTotalSizeBytes(layout);

            uint32_t singleBufferSize = layout.getTotalSizeBytes();
            uint32_t elementCount     = layout.getItemCount();
            uint32_t bufferIndex      = 0;

            for (uint32_t bufferArrayIndex = 0; bufferArrayIndex < bufferCount;
                 bufferArrayIndex++) {
                auto allocationInfo = allocationInfos[bufferArrayIndex];

                for (uint64_t bufferOffset = 0; bufferOffset < bufferTotalSize;
                     bufferOffset += singleBufferSize) {
                    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                    uint8_t* bufferBeginBytesPointer =
                        static_cast<uint8_t*>(allocationInfo.pMappedData) + bufferOffset;
                    uint8_t* bufferEndBytesPointer = bufferBeginBytesPointer + singleBufferSize;
                    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

                    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
                    fillFunction(
                        bufferIndex,
                        reinterpret_cast<layout_type::content_type_pointer>(
                            bufferBeginBytesPointer),
                        reinterpret_cast<layout_type::content_type_pointer>(bufferEndBytesPointer),
                        layout);
                    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
                    bufferIndex++;
                }
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
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT,
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
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                   layoutT>;
} // namespace epseon::gpu::cpp::allocation
