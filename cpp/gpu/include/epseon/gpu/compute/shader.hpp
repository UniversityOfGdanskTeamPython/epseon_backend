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
                devicePointer(devicePtr_),
                scalingPointer(scalingPtr_){};

            Base(const Base&)     = delete;
            Base(Base&&) noexcept = default;

            virtual ~Base() = default;

            Base& operator=(const Base&)     = delete;
            Base& operator=(Base&&) noexcept = default;

            virtual void run() = 0;

          protected:
            std::shared_ptr<environment::Device>& getDevicePointer() {
                return this->devicePointer;
            }

            std::shared_ptr<scaling::Base>& getScalingPointer() {
                return this->scalingPointer;
            }

          private:
            std::shared_ptr<environment::Device> devicePointer  = {};
            std::shared_ptr<scaling::Base>       scalingPointer = {};
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
                resource.prepare(this->getDevicePointer(), this->getScalingPointer());
            }

          private:
            resources::Dynamic resource;
        };

        template <resources::Concept resourceT>
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
                resource.prepare(this->devicePointer, this->scalingPointer);
            }

          private:
            resourceT resource;
        };
    } // namespace shader

    class VibwaResources : public resources::Static {
      public:
        explicit VibwaResources(uint64_t batchSize_) :
            configuration({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 0}}),
            y({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 1}}),
            buffer0({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 2}}),
            output({{.batchSize = batchSize_, .itemCount = 0, .set = 0, .binding = 3}}) {}

      protected:
        template <typename CallableT>
        void forEachBuffer(CallableT callable) {
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
