#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

#include "epseon/gpu/compute/allocation.hpp"
#include "epseon/gpu/compute/buffer.hpp"
#include "epseon/gpu/compute/environment.hpp"
#include "epseon/gpu/compute/layout.hpp"
#include "epseon/gpu/compute/scaling.hpp"
#include "epseon/gpu/compute/structs.hpp"

#include "vk_mem_alloc.h"
#include "vk_mem_alloc_handles.hpp"
#include "vk_mem_alloc_structs.hpp"
#include <functional>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

namespace epseon::gpu::cpp::environment {
    class Device;
}

namespace epseon::gpu::cpp::buffer {
    template <layout::Concept layoutT, allocation::Concept boundAllocationT>
    class Base {
      public:
        Base() = delete;

        Base(layoutT&& layout_) : // NOLINT(hicpp-explicit-conversions)
            layout(layout_){};

        Base(const layoutT& layout_) : // NOLINT(hicpp-explicit-conversions)
            layout(layout_){};

        Base(const Base& other)     = default;
        Base(Base&& other) noexcept = default;

        virtual ~Base() = default;

        Base& operator=(const Base& other)     = default;
        Base& operator=(Base&& other) noexcept = default;

        void bind(std::shared_ptr<environment::Device>& devicePointer,
                  std::shared_ptr<scaling::Base>&       scalingPointer) {
            this->devicePointer  = devicePointer;
            this->scalingPointer = scalingPointer;
        }

        [[nodiscard]] layoutT& getLayout() {
            return this->layout;
        }

        [[nodiscard]] const layoutT& getLayout() const {
            return this->layout;
        }

        [[nodiscard]] scaling::Base& getScaling() {
            assert(scalingPointer);
            return *this->scalingPointer;
        }

        [[nodiscard]] const scaling::Base& getScaling() const {
            assert(scalingPointer);
            return *this->scalingPointer;
        }

        [[nodiscard]] uint32_t getMaxDescriptorSets() const {
            return this->getLayout().getSet();
        }

        virtual void                                  allocateBuffers()      = 0;
        virtual void                                  deallocateBuffers()    = 0;
        [[nodiscard]] virtual boundAllocationT&       getBoundBuffer()       = 0;
        [[nodiscard]] virtual const boundAllocationT& getBoundBuffer() const = 0;
        virtual void recordHostToDeviceTransfers(vk::raii::CommandBuffer&)   = 0;
        virtual void recordDeviceToHostTransfers(vk::raii::CommandBuffer&)   = 0;

        [[nodiscard]] vk::DescriptorPoolSize getDescriptorPoolSize() const {
            return getBoundBuffer().getDescriptorPoolSize(this->getLayout());
        }

        [[nodiscard]] std::optional<vk::DescriptorSetLayoutBinding>
        getDescriptorSetLayoutBindings(uint32_t setIndex) const {
            return getBoundBuffer().getDescriptorSetLayoutBindings(setIndex, this->getLayout());
        }

        [[nodiscard]] DescriptorSetWrite
        getDescriptorSetWrite(vk::raii::DescriptorSet& descriptorSet) {
            auto buffer = getBoundBuffer();

            DescriptorSetWrite writeInfo;
            writeInfo.bufferInfo = getBoundBuffer().getBufferInfo();

            writeInfo.writeInfo.setDstSet(*descriptorSet)
                .setDstBinding(getLayout().getBinding())
                .setBufferInfo(writeInfo.bufferInfo)
                .setDescriptorCount(getBoundBuffer().getDescriptorCount())
                .setDescriptorType(getBoundBuffer().getDescriptorType());

            return writeInfo;
        }

        [[nodiscard]] std::vector<vk::DescriptorBufferInfo> getBufferInfo() const {
            return getBoundBuffer().getBufferInfo();
        }

      private:
        layoutT                              layout;
        std::shared_ptr<environment::Device> devicePointer  = {};
        std::shared_ptr<scaling::Base>       scalingPointer = {};
    };

    template <layout::Concept layoutT>
    class DeviceLocal : public Base<layoutT, allocation::DeviceLocal<layoutT>> {
        using boundAllocationT = allocation::DeviceLocal<layoutT>;

      public:
        DeviceLocal() = delete;

        DeviceLocal(layoutT&& layout_) : // NOLINT(hicpp-explicit-conversions)
            Base<layoutT, boundAllocationT>(layout_){};

        DeviceLocal(const layoutT& layout_) : // NOLINT(hicpp-explicit-conversions)
            Base<layoutT, boundAllocationT>(layout_){};

        DeviceLocal(const DeviceLocal& other)     = default;
        DeviceLocal(DeviceLocal&& other) noexcept = default;

        ~DeviceLocal() override = default;

        DeviceLocal& operator=(const DeviceLocal& other)     = default;
        DeviceLocal& operator=(DeviceLocal&& other) noexcept = default;

        void bind(std::shared_ptr<environment::Device>& devicePointer,
                  std::shared_ptr<scaling::Base>&       scalingPointer) {
            this->devicePointer  = devicePointer;
            this->scalingPointer = scalingPointer;
            this->deviceBuffer.bind(devicePointer, scalingPointer);
        }

        void allocateBuffers() override {
            deviceBuffer.allocateBuffers(this->getLayout());
        }

        void deallocateBuffers() override {
            deviceBuffer.deallocateBuffers(this->getLayout());
        }

        [[nodiscard]] boundAllocationT& getBoundBuffer() override {
            return this->deviceBuffer;
        }

        [[nodiscard]] const boundAllocationT& getBoundBuffer() const override {
            return this->deviceBuffer;
        }

        void recordHostToDeviceTransfers(vk::raii::CommandBuffer& /*commandBuffer*/) override {}

        void recordDeviceToHostTransfers(vk::raii::CommandBuffer& /*commandBuffer*/) override {}

      private:
        allocation::DeviceLocal<layoutT> deviceBuffer = {};
    };

    template <allocation::Concept sourceAllocationT,
              allocation::Concept destinationAllocationT,
              allocation::Concept boundAllocationT,
              allocation::Concept mappedAllocationT,
              layout::Concept     layoutT>
    class Transferable : public Base<layoutT, boundAllocationT> {
        static_assert(
            std::is_same<typename sourceAllocationT::layout_type,
                         typename destinationAllocationT::layout_type>::value,
            "sourceBufferT::layout_type and sourceBufferT::layout_type must be of the same "
            "type.");
        static_assert(std::is_same<typename sourceAllocationT::layout_type, layoutT>::value,
                      "sourceBufferT::layout_type and layoutT must be of the same "
                      "type.");

      public:
        Transferable() = delete;

        Transferable(layoutT&& layout_) : // NOLINT(hicpp-explicit-conversions)
            Base<layoutT, boundAllocationT>(layout_){};

        Transferable(const layoutT& layout_) : // NOLINT(hicpp-explicit-conversions)
            Base<layoutT, boundAllocationT>(layout_){};

        Transferable(const Transferable& other)     = default;
        Transferable(Transferable&& other) noexcept = default;

        ~Transferable() override = default;

        Transferable& operator=(const Transferable& other)     = default;
        Transferable& operator=(Transferable&& other) noexcept = default;

        void bind(std::shared_ptr<environment::Device>& devicePointer,
                  std::shared_ptr<scaling::Base>&       scalingPointer) {
            this->devicePointer  = devicePointer;
            this->scalingPointer = scalingPointer;
            this->sourceBuffer.bind(devicePointer, scalingPointer);
            this->destinationBuffer.bind(devicePointer, scalingPointer);
        }

        void allocateBuffers() override {
            sourceBuffer.allocateBuffers(this->getLayout());
            destinationBuffer.allocateBuffers(this->getLayout());
        }

        void deallocateBuffers() override {
            sourceBuffer.deallocateBuffers(this->getLayout());
            destinationBuffer.deallocateBuffers(this->getLayout());
        }

        sourceAllocationT& getSourceAllocation() {
            return this->sourceBuffer;
        }

        const sourceAllocationT& getSourceAllocation() const {
            return this->sourceBuffer;
        }

        destinationAllocationT& getDestinationAllocation() {
            return this->destinationBuffer;
        }

        const destinationAllocationT& getDestinationAllocation() const {
            return this->destinationBuffer;
        }

        virtual mappedAllocationT& getMappedAllocation() = 0;

        virtual const mappedAllocationT& getMappedAllocation() const = 0;

        void fillBuffers(layoutT::fill_function_type fillFunction) {
            getMappedAllocation().fillBuffers(fillFunction, this->getLayout());
        }

      private:
        sourceAllocationT      sourceBuffer      = {};
        destinationAllocationT destinationBuffer = {};
    };

    template <layout::Concept layoutT>
    class HostToDevice : public Transferable<allocation::HostTransferSrc<layoutT>,
                                             allocation::DeviceTransferDst<layoutT>,
                                             allocation::DeviceTransferDst<layoutT>,
                                             allocation::HostTransferSrc<layoutT>,
                                             layoutT> {
        using boundAllocationT  = allocation::DeviceTransferDst<layoutT>;
        using mappedAllocationT = allocation::HostTransferSrc<layoutT>;
        using transferableT     = Transferable<allocation::HostTransferSrc<layoutT>,
                                           allocation::DeviceTransferDst<layoutT>,
                                           allocation::DeviceTransferDst<layoutT>,
                                           allocation::HostTransferSrc<layoutT>,
                                           layoutT>;

      public:
        HostToDevice() = delete;

        HostToDevice(layoutT&& layout_) : // NOLINT(hicpp-explicit-conversions)
            transferableT(layout_){};

        HostToDevice(const layoutT& layout_) : // NOLINT(hicpp-explicit-conversions)
            transferableT(layout_){};

        HostToDevice(const HostToDevice& other)     = default;
        HostToDevice(HostToDevice&& other) noexcept = default;

        virtual ~HostToDevice() = default;

        HostToDevice& operator=(const HostToDevice& other)     = default;
        HostToDevice& operator=(HostToDevice&& other) noexcept = default;

        [[nodiscard]] boundAllocationT& getBoundBuffer() override {
            return this->getDestinationAllocation();
        }

        [[nodiscard]] const boundAllocationT& getBoundBuffer() const override {
            return this->getDestinationAllocation();
        }

        mappedAllocationT& getMappedAllocation() override {
            return this->getSourceAllocation();
        }

        const mappedAllocationT& getMappedAllocation() const override {
            return this->getSourceAllocation();
        }

        void recordHostToDeviceTransfers(vk::raii::CommandBuffer& commandBuffer) override {
            this->getSourceAllocation().recordCopyBuffer(
                this->getDestinationAllocation(), commandBuffer, this->getLayout());
        }

        void recordDeviceToHostTransfers(vk::raii::CommandBuffer& /*commandBuffer*/) override {}
    };

    template <layout::Concept layoutT>
    class DeviceToHost : public Transferable<allocation::DeviceTransferSrc<layoutT>,
                                             allocation::HostTransferDst<layoutT>,
                                             allocation::DeviceTransferSrc<layoutT>,
                                             allocation::HostTransferDst<layoutT>,
                                             layoutT> {
        using boundAllocationT  = allocation::DeviceTransferSrc<layoutT>;
        using mappedAllocationT = allocation::HostTransferDst<layoutT>;
        using transferableT     = Transferable<allocation::DeviceTransferSrc<layoutT>,
                                           allocation::HostTransferDst<layoutT>,
                                           allocation::DeviceTransferSrc<layoutT>,
                                           allocation::HostTransferDst<layoutT>,
                                           layoutT>;

      public:
        DeviceToHost() = delete;

        DeviceToHost(layoutT&& layout_) : // NOLINT(hicpp-explicit-conversions)
            transferableT(layout_){};

        DeviceToHost(const layoutT& layout_) : // NOLINT(hicpp-explicit-conversions)
            transferableT(layout_){};

        DeviceToHost(const DeviceToHost& other)     = default;
        DeviceToHost(DeviceToHost&& other) noexcept = default;

        virtual ~DeviceToHost() = default;

        DeviceToHost& operator=(const DeviceToHost& other)     = default;
        DeviceToHost& operator=(DeviceToHost&& other) noexcept = default;

        [[nodiscard]] boundAllocationT& getBoundBuffer() override {
            return this->getSourceAllocation();
        }

        [[nodiscard]] const boundAllocationT& getBoundBuffer() const override {
            return this->getSourceAllocation();
        }

        mappedAllocationT& getMappedAllocation() override {
            return this->getDestinationAllocation();
        }

        const mappedAllocationT& getMappedAllocation() const override {
            return this->getDestinationAllocation();
        }

        void recordHostToDeviceTransfers(vk::raii::CommandBuffer& /*commandBuffer*/) override {}

        void recordDeviceToHostTransfers(vk::raii::CommandBuffer& commandBuffer) override {
            this->getSourceAllocation().recordCopyBuffer(
                this->getDestinationAllocation(), commandBuffer, this->getLayout());
        }
    };

    namespace dynamic {
        using HostToDevice = buffer::HostToDevice<layout::Dynamic>;
        using DeviceLocal  = buffer::DeviceLocal<layout::Dynamic>;
        using DeviceToHost = buffer::DeviceToHost<layout::Dynamic>;
    } // namespace dynamic

} // namespace epseon::gpu::cpp::buffer