#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

#include "epseon/gpu/compute/allocation.hpp"
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
    template <layout::Concept layoutT>
    class Base {
      public:
        Base() = default;

        explicit Base(layoutT&& layout_) :
            layout(layout_){};

        explicit Base(const layoutT& layout_) :
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

        [[nodiscard]] uint32_t getMaxDescriptorSets() const {
            return this->getLayout().getSet();
        }

        virtual void                                 allocateBuffers()             = 0;
        virtual void                                 deallocateBuffers()           = 0;
        [[nodiscard]] virtual vk::DescriptorPoolSize getDescriptorPoolSize() const = 0;
        [[nodiscard]] virtual std::optional<vk::DescriptorSetLayoutBinding>
        getDescriptorSetLayoutBindings(uint32_t setIndex) const = 0;

      private:
        layoutT                              layout         = {};
        std::shared_ptr<environment::Device> devicePointer  = {};
        std::shared_ptr<scaling::Base>       scalingPointer = {};
    };

    template <layout::Concept layoutT>
    class DeviceLocal : public Base<layoutT> {
      public:
        using Base<layoutT>::Base;

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

        [[nodiscard]] vk::DescriptorPoolSize getDescriptorPoolSize() const override {
            return deviceBuffer.getDescriptorPoolSize(this->getLayout());
        }

        [[nodiscard]] std::optional<vk::DescriptorSetLayoutBinding>
        getDescriptorSetLayoutBindings(uint32_t setIndex) const override {
            return deviceBuffer.getDescriptorSetLayoutBindings(setIndex, this->getLayout());
        }

      private:
        allocation::DeviceLocal<layoutT> deviceBuffer = {};
    };

    template <typename sourceBufferT, typename destinationBufferT, layout::Concept layoutT>
    class Transferable : public Base<layoutT> {
        static_assert(
            std::is_same<typename sourceBufferT::layout_type,
                         typename destinationBufferT::layout_type>::value,
            "sourceBufferT::layout_type and sourceBufferT::layout_type must be of the same "
            "type.");
        static_assert(std::is_same<typename sourceBufferT::layout_type, layoutT>::value,
                      "sourceBufferT::layout_type and layoutT must be of the same "
                      "type.");

      public:
        using Base<layoutT>::Base;

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

    template <layout::Concept layoutT>
    class HostToDevice : public Transferable<allocation::HostTransferSrc<layoutT>,
                                             allocation::DeviceTransferDst<layoutT>,
                                             layoutT> {
      public:
        using Transferable<allocation::HostTransferSrc<layoutT>,
                           allocation::DeviceTransferDst<layoutT>,
                           layoutT>::Transferable;

        [[nodiscard]] vk::DescriptorPoolSize getDescriptorPoolSize() const override {
            return this->getSourceBuffer().getDescriptorPoolSize(this->getLayout());
        }

        [[nodiscard]] std::optional<vk::DescriptorSetLayoutBinding>
        getDescriptorSetLayoutBindings(uint32_t setIndex) const override {
            return this->getSourceBuffer().getDescriptorSetLayoutBindings(setIndex,
                                                                          this->getLayout());
        }
    };

    template <layout::Concept layoutT>
    class DeviceToHost : public Transferable<allocation::DeviceTransferSrc<layoutT>,
                                             allocation::HostTransferDst<layoutT>,
                                             layoutT> {
      public:
        using Transferable<allocation::DeviceTransferSrc<layoutT>,
                           allocation::HostTransferDst<layoutT>,
                           layoutT>::Transferable;

        [[nodiscard]] vk::DescriptorPoolSize getDescriptorPoolSize() const override {
            return this->getDestinationBuffer().getDescriptorPoolSize(this->getLayout());
        }

        [[nodiscard]] std::optional<vk::DescriptorSetLayoutBinding>
        getDescriptorSetLayoutBindings(uint32_t setIndex) const override {
            return this->getDestinationBuffer().getDescriptorSetLayoutBindings(setIndex,
                                                                               this->getLayout());
        }
    };

    namespace dynamic {
        using HostToDevice = buffer::HostToDevice<layout::Dynamic>;
        using DeviceLocal  = buffer::DeviceLocal<layout::Dynamic>;
        using DeviceToHost = buffer::DeviceToHost<layout::Dynamic>;
    } // namespace dynamic

} // namespace epseon::gpu::cpp::buffer