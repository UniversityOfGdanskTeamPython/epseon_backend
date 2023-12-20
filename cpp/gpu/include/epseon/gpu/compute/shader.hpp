#pragma once

#include "epseon/gpu/compute/spirv.hpp"
#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

#include "epseon/gpu/common.hpp"
#include "epseon/gpu/compute/allocation.hpp"
#include "epseon/gpu/compute/buffer.hpp"
#include "epseon/gpu/compute/environment.hpp"
#include "epseon/gpu/compute/layout.hpp"
#include "epseon/gpu/compute/resources.hpp"
#include "epseon/gpu/compute/scaling.hpp"
#include "epseon/gpu/compute/structs.hpp"

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

            void prepare(GLSL& sourceCode) {
                auto bytecode = sourceCode.compile();
                prepare(bytecode);
            }

            void prepare(SPIRV& bytecode) {
                auto& resource = getResource();
                resource.prepare(bytecode, this->getDevicePointer(), this->getScalingPointer());
            }

            void run() {}

            [[nodiscard]] virtual resourceT&       getResource()       = 0;
            [[nodiscard]] virtual const resourceT& getResource() const = 0;

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
            using resourceT = resources::Dynamic;

          public:
            Dynamic() = delete;

            Dynamic(resourceT&&                           resource_,
                    std::shared_ptr<environment::Device>& devicePtr_,
                    std::shared_ptr<scaling::Base>&       scalingPtr_) :
                Base<resources::Dynamic>(devicePtr_, scalingPtr_),
                resource(std::move(resource_)){};

            Dynamic(const Dynamic&)     = delete;
            Dynamic(Dynamic&&) noexcept = default;

            ~Dynamic() override = default;

            Dynamic& operator=(const Dynamic&)     = delete;
            Dynamic& operator=(Dynamic&&) noexcept = default;

            [[nodiscard]] resourceT& getResource() override {
                return this->resource;
            }

            [[nodiscard]] const resourceT& getResource() const override {
                return this->resource;
            }

          private:
            resourceT resource;
        };

        template <resources::Concept resourceT>
        class Static : public Base<resourceT> {
          public:
            Static() = delete;

            Static(resourceT&&                           resource_,
                   std::shared_ptr<environment::Device>& devicePtr_,
                   std::shared_ptr<scaling::Base>&       scalingPtr_) :
                Base<resourceT>(devicePtr_, scalingPtr_),
                resource(std::move(resource_)){};

            Static(const Static&)     = delete;
            Static(Static&&) noexcept = default;

            ~Static() override = default;

            Static& operator=(const Static&)     = delete;
            Static& operator=(Static&&) noexcept = default;

            [[nodiscard]] resourceT& getResource() override {
                return this->resource;
            }

            [[nodiscard]] const resourceT& getResource() const override {
                return this->resource;
            }

          private:
            resourceT resource;
        };
    } // namespace shader
} // namespace epseon::gpu::cpp
