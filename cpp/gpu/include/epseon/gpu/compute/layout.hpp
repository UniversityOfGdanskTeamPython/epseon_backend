#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

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
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace epseon::gpu::cpp::layout {

    class Base {
      public:
        Base() = delete;

        explicit Base(uint64_t batchSize_, uint32_t set_, uint32_t binding_) :
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

        [[nodiscard]] uint32_t getSet() const {
            return this->set;
        };

        [[nodiscard]] uint32_t getBinding() const {
            return this->binding;
        };

      private:
        uint64_t batchSize = {};
        uint32_t set       = {};
        uint32_t binding   = {};
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
} // namespace epseon::gpu::cpp::layout