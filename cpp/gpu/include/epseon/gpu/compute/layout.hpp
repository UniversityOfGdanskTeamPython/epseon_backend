#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

#include "epseon/gpu/compute/scaling.hpp"
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
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace epseon::gpu::cpp::layout {

    class Base {
      public:
        using content_type         = void;
        using content_type_pointer = void*;
        using fill_function_type =
            std::function<void(uint32_t, content_type_pointer, content_type_pointer, const Base&)>;

        Base() = delete;

        Base(uint32_t set_, uint32_t binding_) :
            set(set_),
            binding(binding_){};

        Base(const Base& other)     = default;
        Base(Base&& other) noexcept = default;

        virtual ~Base() = default;

        Base& operator=(const Base& other)     = default;
        Base& operator=(Base&& other) noexcept = default;

        [[nodiscard]] virtual uint64_t getItemCount() const = 0;
        [[nodiscard]] virtual uint64_t getItemSize() const  = 0;

        [[nodiscard]] uint64_t getTotalSizeBytes() const {
            return this->getItemCount() * this->getItemSize();
        };

        [[nodiscard]] uint32_t getSet() const {
            return this->set;
        };

        [[nodiscard]] uint32_t getBinding() const {
            return this->binding;
        };

      private:
        uint32_t set     = {};
        uint32_t binding = {};
    };

    template <typename contentT>
    class Static : public Base {
      public:
        using content_type         = contentT;
        using content_type_pointer = contentT*;
        using fill_function_type   = std::function<void(
            uint32_t, content_type_pointer, content_type_pointer, const Static<content_type>&)>;

        Static() = delete;

        struct StaticCtorParams {
            uint64_t itemCount;
            uint64_t set;
            uint64_t binding;
        };

        Static(StaticCtorParams params) : // NOLINT(hicpp-explicit-conversions)
            Base(params.set, params.binding),
            itemCount(params.itemCount) {}

        Static(uint64_t itemCount, uint64_t set, uint64_t binding) :
            Base(set, binding),
            itemCount(itemCount) {}

        Static(const Static& other)     = default;
        Static(Static&& other) noexcept = default;

        ~Static() override = default;

        Static& operator=(const Static& other)     = default;
        Static& operator=(Static&& other) noexcept = default;

        [[nodiscard]] uint64_t getItemCount() const override {
            return itemCount;
        };

        [[nodiscard]] uint64_t getItemSize() const override {
            return sizeof(content_type);
        };

      private:
        uint64_t itemCount = {};
    };

    class Dynamic : public Base {
      public:
        using content_type         = void;
        using content_type_pointer = void*;
        using fill_function_type   = std::function<void(
            uint32_t, content_type_pointer, content_type_pointer, const Dynamic&)>;

        Dynamic() = delete;

        struct DynamicCtorParams {
            uint64_t itemCount;
            uint64_t itemSize;
            uint64_t set;
            uint64_t binding;
        };

        Dynamic(DynamicCtorParams params) : // NOLINT(hicpp-explicit-conversions)
            Base(params.set, params.binding),
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
} // namespace epseon::gpu::cpp::layout