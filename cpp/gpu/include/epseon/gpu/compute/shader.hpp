#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/algorithms/vibwa.hpp"
#include "vk_mem_alloc.h"
#include "vk_mem_alloc_handles.hpp"
#include "vk_mem_alloc_structs.hpp"
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace epseon::gpu::cpp {

    namespace scaling {

        template <class T>
        concept Concept = requires(T t, uint64_t uint64) {
            { T::allocation::getTotalSizeBytes(uint64, uint64) } -> std::same_as<uint64_t>;
            { T::allocation::getBufferCount(uint64) } -> std::same_as<uint64_t>;
        };

        class Base {};

        class BufferArray : Base {
          public:
            struct allocation {
                [[nodiscard]] static uint64_t getTotalSizeBytes(uint64_t totalSizeBytes,
                                                                uint64_t /*batchSize*/) {
                    return totalSizeBytes;
                }

                [[nodiscard]] static uint64_t getBufferCount(uint64_t batchSize) {
                    return batchSize;
                }
            };
        };

        class LargeBuffer : Base {
          public:
            struct allocation {
                [[nodiscard]] static uint64_t getTotalSizeBytes(uint64_t totalSizeBytes,
                                                                uint64_t batchSize) {
                    return totalSizeBytes * batchSize;
                }

                [[nodiscard]] static uint64_t getBufferCount(uint64_t /*batchSize*/) {
                    return 1;
                }
            };
        };
    }; // namespace scaling

    namespace layout {

        template <typename T>
        concept Concept = requires(T t) {
            { t.getItemCount() } -> std::same_as<uint64_t>;
            { t.getItemSize() } -> std::same_as<uint64_t>;
            { t.getBatchSize() } -> std::same_as<uint64_t>;
            { t.getTotalSizeBytes() } -> std::same_as<uint64_t>;
        };

        class Base {
          public:
            Base() = delete;

            explicit Base(uint64_t batchSize_, uint64_t set_, uint64_t binding_) :
                batchSize(batchSize_),
                set(set_),
                binding(binding_){};

            Base(const Base& other)     = default;
            Base(Base&& other) noexcept = default;

            virtual ~Base() = default;

            Base& operator=(const Base& other)     = default;
            Base& operator=(Base&& other) noexcept = default;

            [[nodiscard]] virtual uint64_t getItemCount() const = 0;
            [[nodiscard]] virtual uint64_t getItemSize() const  = 0;

            [[nodiscard]] uint64_t getBatchSize() const {
                return this->batchSize;
            }

            [[nodiscard]] uint64_t getTotalSizeBytes() const {
                return this->getItemCount() * this->getItemSize();
            };

          private:
            uint64_t batchSize = {};
            uint64_t set       = {};
            uint64_t binding   = {};
        };

        template <typename contentT>
        class Static : public Base {
          public:
            Static() = delete;

            struct StaticCtorParams {
                uint64_t batchSize;
                uint64_t itemCount;
                uint64_t set;
                uint64_t binding;
            };

            Static(StaticCtorParams params) : // NOLINT(hicpp-explicit-conversions)
                Base(params.batchSize, params.set, params.binding),
                itemCount(params.itemCount) {}

            Static(const Static& other)     = default;
            Static(Static&& other) noexcept = default;

            ~Static() override = default;

            Static& operator=(const Static& other)     = default;
            Static& operator=(Static&& other) noexcept = default;

            [[nodiscard]] uint64_t getItemCount() const override {
                return itemCount;
            };

            [[nodiscard]] uint64_t getItemSize() const override {
                return sizeof(contentT);
            };

          private:
            uint64_t itemCount = {};
        };

        class Dynamic : public Base {
          public:
            Dynamic() = delete;

            struct DynamicCtorParams {
                uint64_t batchSize;
                uint64_t itemCount;
                uint64_t itemSize;
                uint64_t set;
                uint64_t binding;
            };

            Dynamic(DynamicCtorParams params) : // NOLINT(hicpp-explicit-conversions)
                Base(params.batchSize, params.set, params.binding),
                itemCount(params.itemCount),
                itemSize(params.itemSize) {}

            Dynamic(const Dynamic& other)     = default;
            Dynamic(Dynamic&& other) noexcept = default;

            ~Dynamic() override = default;

            Dynamic& operator=(const Dynamic& other)     = default;
            Dynamic& operator=(Dynamic&& other) noexcept = default;

            [[nodiscard]] uint64_t getItemCount() const override {
                return itemCount;
            };

            [[nodiscard]] uint64_t getItemSize() const override {
                return itemSize;
            };

          private:
            uint64_t itemCount = {};
            uint64_t itemSize  = {};
        };
    } // namespace layout

    namespace allocation {
        template <VkBufferUsageFlags       bufferUsageFlags,
                  VmaAllocationCreateFlags allocationFlags,
                  layout::Concept          layoutT,
                  scaling::Concept         scalingT>
        struct Allocation {

            using layout_type  = layoutT;
            using scaling_type = scalingT;

            std::vector<vk::Buffer>          buffers         = {};
            std::vector<vma::Allocation>     allocations     = {};
            std::vector<vma::AllocationInfo> allocationInfos = {};

            std::vector<vk::Buffer>& getBuffers() {
                return this->buffers;
            }

            std::vector<vma::Allocation>& getAllocations() {
                return this->allocations;
            }

            std::vector<vma::AllocationInfo>& getAllocationInfos() {
                return this->allocationInfos;
            }

            [[nodiscard]] static constexpr vk::BufferUsageFlags getBufferUsageFlags() {
                return static_cast<vk::BufferUsageFlags>(bufferUsageFlags);
            }

            [[nodiscard]] static constexpr vma::AllocationCreateFlags getAllocationFlags() {
                return static_cast<vma::AllocationCreateFlags>(allocationFlags);
            }

            void allocateBuffers(const vma::Allocator& allocator, const layoutT& layout) {
                auto bufferSizeBytes = scalingT::allocation::getTotalSizeBytes(
                    layout.getTotalSizeBytes(), layout.getBatchSize());
                auto bufferCreateInfo =
                    vk::BufferCreateInfo().setUsage(getBufferUsageFlags()).setSize(bufferSizeBytes);
                auto allocationCreateInfo = vma::AllocationCreateInfo()
                                                .setUsage(vma::MemoryUsage::eAuto)
                                                .setFlags(getAllocationFlags());

                auto bufferCount = scalingT::allocation::getBufferCount(layout.getBatchSize());

                buffers.reserve(bufferCount);
                allocations.reserve(bufferCount);
                allocationInfos.reserve(bufferCount);

                for (uint64_t i = 0; i < bufferCount; i++) {
                    vma::AllocationInfo info{};

                    auto [buffer, allocation] =
                        allocator.createBuffer(bufferCreateInfo, allocationCreateInfo, &info);

                    buffers.push_back(std::move(buffer));
                    allocations.push_back(std::move(allocation));
                    allocationInfos.push_back(info);
                }
            }

            void deallocateBuffers(const vma::Allocator& allocator, const layoutT& layout) {
                auto bufferCount = scalingT::allocation::getBufferCount(layout.getBatchSize());

                for (uint32_t i = 0; i < bufferCount; i++) {
                    allocator.destroyBuffer(buffers[i], allocations[i]);
                }
                buffers.clear();
                allocations.clear();
                allocationInfos.clear();
            }

            // vk::DescriptorSetLayoutBinding getDescriptorSetLayoutBindings(const layoutT& layout)
            // {}

            vk::DescriptorPoolSize getDescriptorPoolSize(const layoutT& layout) {
                return vk::DescriptorPoolSize()
                    .setType(vk::DescriptorType::eStorageBuffer)
                    .setDescriptorCount(
                        scalingT::allocation::getBufferCount(layout.getBatchSize()));
            }
        };

        template <layout::Concept layoutT, scaling::Concept scalingT>
        using HostTransferSrc =
            Allocation<VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                       layoutT,
                       scalingT>;

        template <layout::Concept layoutT, scaling::Concept scalingT>
        using DeviceTransferDst =
            Allocation<VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                       layoutT,
                       scalingT>;

        template <layout::Concept layoutT, scaling::Concept scalingT>
        using DeviceLocal = Allocation<VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                       layoutT,
                                       scalingT>;

        template <layout::Concept layoutT, scaling::Concept scalingT>
        using DeviceTransferSrc =
            Allocation<VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                       layoutT,
                       scalingT>;

        template <layout::Concept layoutT, scaling::Concept scalingT>
        using HostTransferDst =
            Allocation<VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                       layoutT,
                       scalingT>;
    } // namespace allocation

    namespace buffer {
        template <layout::Concept layoutT>
        class Base {
          public:
            Base() = default;

            Base(layoutT&& layout_) : // NOLINT(hicpp-explicit-conversions)
                layout(layout_){};

            Base(const layoutT& layout_) : // NOLINT(hicpp-explicit-conversions)
                layout(layout_){};

            Base(const Base& other)     = default;
            Base(Base&& other) noexcept = default;

            virtual ~Base() = default;

            Base& operator=(const Base& other)     = default;
            Base& operator=(Base&& other) noexcept = default;

            void bindDeviceAllocator(std::shared_ptr<vma::raii::Allocator>& allocator) {
                this->allocator = allocator;
            }

            vma::Allocator& getDeviceAllocator() {
                return *this->allocator;
            }

            layoutT& getLayout() {
                return this->layout;
            }

            virtual void                   allocateBuffers()       = 0;
            virtual void                   deallocateBuffers()     = 0;
            virtual vk::DescriptorPoolSize getDescriptorPoolSize() = 0;

          private:
            layoutT                               layout    = {};
            std::shared_ptr<vma::raii::Allocator> allocator = {};
        };

        template <layout::Concept layoutT, scaling::Concept scalingT>
        class DeviceLocal : public Base<layoutT> {
          public:
            using Base<layoutT>::Base;

            void allocateBuffers() override {
                deviceBuffer.allocateBuffers(this->getDeviceAllocator(), this->getLayout());
            }

            void deallocateBuffers() override {
                deviceBuffer.deallocateBuffers(this->getDeviceAllocator(), this->getLayout());
            }

            vk::DescriptorPoolSize getDescriptorPoolSize() override {
                return deviceBuffer.getDescriptorPoolSize(this->getLayout());
            }

          private:
            allocation::DeviceLocal<layoutT, scalingT> deviceBuffer = {};
        };

        template <typename sourceBufferT,
                  typename destinationBufferT,
                  layout::Concept  layoutT,
                  scaling::Concept scalingT>
        class Transferable : public Base<layoutT> {
            static_assert(
                std::is_same<typename sourceBufferT::layout_type,
                             typename destinationBufferT::layout_type>::value,
                "sourceBufferT::layout_type and sourceBufferT::layout_type must be of the same "
                "type.");
            static_assert(std::is_same<typename sourceBufferT::layout_type, layoutT>::value,
                          "sourceBufferT::layout_type and layoutT must be of the same "
                          "type.");

            static_assert(
                std::is_same<typename sourceBufferT::scaling_type,
                             typename destinationBufferT::scaling_type>::value,
                "sourceBufferT::scaling_type and sourceBufferT::scaling_type must be of the same "
                "type.");
            static_assert(std::is_same<typename sourceBufferT::scaling_type, scalingT>::value,
                          "sourceBufferT::scaling_type and scalingT must be of the same "
                          "type.");

          public:
            using Base<layoutT>::Base;

            void allocateBuffers() override {
                sourceBuffer.allocateBuffers(this->getDeviceAllocator(), this->getLayout());
                destinationBuffer.allocateBuffers(this->getDeviceAllocator(), this->getLayout());
            }

            void deallocateBuffers() override {
                sourceBuffer.deallocateBuffers(this->getDeviceAllocator(), this->getLayout());
                destinationBuffer.deallocateBuffers(this->getDeviceAllocator(), this->getLayout());
            }

            sourceBufferT& getSourceBuffer() {
                return this->sourceBuffer;
            }

            destinationBufferT& getDestinationBuffer() {
                return this->destinationBuffer;
            }

          private:
            sourceBufferT      sourceBuffer      = {};
            destinationBufferT destinationBuffer = {};
        };

        template <layout::Concept layoutT, scaling::Concept scalingT>
        class HostToDevice : public Transferable<allocation::HostTransferSrc<layoutT, scalingT>,
                                                 allocation::DeviceTransferDst<layoutT, scalingT>,
                                                 layoutT,
                                                 scalingT> {
          public:
            using Transferable<allocation::HostTransferSrc<layoutT, scalingT>,
                               allocation::DeviceTransferDst<layoutT, scalingT>,
                               layoutT,
                               scalingT>::Transferable;

            vk::DescriptorPoolSize getDescriptorPoolSize() override {
                return this->getSourceBuffer().getDescriptorPoolSize(this->getLayout());
            }
        };

        template <layout::Concept layoutT, scaling::Concept scalingT>
        class DeviceToHost : public Transferable<allocation::DeviceTransferSrc<layoutT, scalingT>,
                                                 allocation::HostTransferDst<layoutT, scalingT>,
                                                 layoutT,
                                                 scalingT> {
          public:
            using Transferable<allocation::DeviceTransferSrc<layoutT, scalingT>,
                               allocation::HostTransferDst<layoutT, scalingT>,
                               layoutT,
                               scalingT>::Transferable;

            vk::DescriptorPoolSize getDescriptorPoolSize() override {
                return this->getDestinationBuffer().getDescriptorPoolSize(this->getLayout());
            }
        };

        namespace dynamic {
            template <scaling::Concept scalingT>
            using HostToDevice = buffer::HostToDevice<layout::Dynamic, scalingT>;

            template <scaling::Concept scalingT>
            using DeviceLocal = buffer::DeviceLocal<layout::Dynamic, scalingT>;

            template <scaling::Concept scalingT>
            using DeviceToHost = buffer::DeviceToHost<layout::Dynamic, scalingT>;
        } // namespace dynamic

    } // namespace buffer

    namespace resources {

        template <typename T>
        concept Concept = requires(T t, std::shared_ptr<vma::raii::Allocator>& allocatorT) {
            { t.bindDeviceAllocator(allocatorT) } -> std::same_as<void>;
        };

        template <scaling::Concept scalingT>
        class Base {
          public:
            Base()                      = default;
            Base(const Base& other)     = delete;
            Base(Base&& other) noexcept = default;

            virtual ~Base() = default;

            Base& operator=(const Base& other)     = delete;
            Base& operator=(Base&& other) noexcept = default;

            void bindDeviceAllocator(std::shared_ptr<vma::raii::Allocator>& allocator) {
                iterateBuffers([&allocator](auto& buffer) {
                    buffer.bindDeviceAllocator(allocator);
                });
            };

            [[nodiscard]] std::vector<vk::DescriptorPoolSize> getDescriptorPoolSize() const {
                std::vector<vk::DescriptorPoolSize> poolSize{};
                uint64_t                            total = 0;
                iterateBuffers([&total](auto& /*buffer*/) {
                    total += 1;
                });

                poolSize.reserve(total);

                iterateBuffers([&poolSize](auto& buffer) {
                    poolSize.push_back(buffer.getDescriptorPoolSize());
                });

                return poolSize;
            };

            template <typename CallableT>
            void iterateBuffers(CallableT /*callable*/) {
                throw std::runtime_error("iterateBuffers must be implemented.");
            }

          private:
            std::unique_ptr<vk::raii::DescriptorPool> descriptorPool = {};
        };

        template <scaling::Concept scalingT>
        class Static : public Base<scalingT> {
          public:
            using Base<scalingT>::Base;
        };

        template <scaling::Concept scalingT>
        class Dynamic : public Base<scalingT> {
          public:
            Dynamic(std::vector<layout::Dynamic>&& inputBuffers_,
                    std::vector<layout::Dynamic>&& deviceLocalBuffers_,
                    std::vector<layout::Dynamic>&& outputBuffers_) {
                inputBuffers.reserve(inputBuffers_.size());
                for (auto& layout : inputBuffers_) {
                    inputBuffers.push_back(buffer::HostToDevice<layout::Dynamic, scalingT>(layout));
                }
                deviceLocalBuffers.reserve(deviceLocalBuffers_.size());
                for (auto& layout : deviceLocalBuffers_) {
                    deviceLocalBuffers.push_back(
                        buffer::DeviceLocal<layout::Dynamic, scalingT>(layout));
                }
                outputBuffers.reserve(outputBuffers_.size());
                for (auto& layout : outputBuffers_) {
                    outputBuffers.push_back(
                        buffer::DeviceToHost<layout::Dynamic, scalingT>(layout));
                }
            }

            template <typename CallableT>
            void iterateBuffers(CallableT callable) {
                for (auto& buffer : inputBuffers) {
                    callable(buffer);
                }
                for (auto& buffer : deviceLocalBuffers) {
                    callable(buffer);
                }
                for (auto& buffer : outputBuffers) {
                    callable(buffer);
                }
            }

          private:
            std::vector<buffer::HostToDevice<layout::Dynamic, scalingT>> inputBuffers       = {};
            std::vector<buffer::DeviceLocal<layout::Dynamic, scalingT>>  deviceLocalBuffers = {};
            std::vector<buffer::DeviceToHost<layout::Dynamic, scalingT>> outputBuffers      = {};
        };
    } // namespace resources

    namespace shader {

        template <resources::Concept resourceT>
        class Base {

          public:
            Base() = default;

            Base(resourceT&& resource_) : // NOLINT(hicpp-explicit-conversions)
                resource(std::move(resource_)){};
            Base(resourceT& resource_) : // NOLINT(hicpp-explicit-conversions)
                resource(resource_){};

            Base(const Base&)     = delete;
            Base(Base&&) noexcept = default;

            ~Base() = default;

            Base& operator=(const Base&)     = delete;
            Base& operator=(Base&&) noexcept = default;

          private:
            resourceT resource = {};
        };

        template <scaling::Concept scalingT>
        using Dynamic = Base<resources::Dynamic<scalingT>>;

        template <resources::Concept resourceT>
        using Static = Base<resourceT>;
    } // namespace shader

    template <scaling::Concept scalingT>
    class VibwaResources : public resources::Static<scalingT> {
      public:
        using resources::Static<scalingT>::Static;

        VibwaResources(uint64_t batchSize_) : // NOLINT(hicpp-explicit-conversions)
            configuration({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 0}}),
            y({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 1}}),
            buffer0({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 1}}),
            output({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 1}}) {}

        template <typename CallableT>
        void iterateBuffers(CallableT callable) {
            for (auto& buffer : configuration) {
                callable(buffer);
            }
            for (auto& buffer : y) {
                callable(buffer);
            }
            for (auto& buffer : buffer0) {
                callable(buffer);
            }
            for (auto& buffer : output) {
                callable(buffer);
            }
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

        buffer::HostToDevice<layout::Static<Configuration>, scalingT> configuration;
        buffer::HostToDevice<layout::Static<float>, scalingT>         y;
        buffer::DeviceLocal<layout::Static<float>, scalingT>          buffer0;
        buffer::DeviceToHost<layout::Static<float>, scalingT>         output;
    };

    template <typename T>
    void foo() { // NOLINT(misc-definitions-in-headers)
        const uint64_t batchSize  = 128;
        const uint64_t bufferSize = 1024;

        shader::Dynamic<scaling::BufferArray> shader1{{{{{.batchSize = batchSize,
                                                          .itemCount = bufferSize,
                                                          .itemSize  = sizeof(float),
                                                          .set       = 0,
                                                          .binding   = 0}}},
                                                       {},
                                                       {}}};

        shader::Static<VibwaResources<scaling::BufferArray>> shader2{{batchSize}};
    }

} // namespace epseon::gpu::cpp
