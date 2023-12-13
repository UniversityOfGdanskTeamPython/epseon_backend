#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

#include "epseon/gpu/compute/allocation.hpp"
#include "epseon/gpu/compute/buffer.hpp"
#include "epseon/gpu/compute/environment.hpp"
#include "epseon/gpu/compute/layout.hpp"
#include "epseon/gpu/compute/scaling.hpp"

#include "vk_mem_alloc.h"
#include "vk_mem_alloc_handles.hpp"
#include "vk_mem_alloc_structs.hpp"
#include <algorithm>
#include <cassert>
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
#include <utility>
#include <vector>

namespace epseon::gpu::cpp::resources {

    class Base {
      public:
        Base()                      = default;
        Base(const Base& other)     = delete;
        Base(Base&& other) noexcept = default;

        virtual ~Base() {
            deallocateBuffers();
        }

        Base& operator=(const Base& other)     = delete;
        Base& operator=(Base&& other) noexcept = default;

        void prepare(std::shared_ptr<environment::Device>& devicePointer,
                     std::shared_ptr<scaling::Base>&       scalingPointer) {
            bind(devicePointer, scalingPointer);
            allocateBuffers();
            createDescriptorPool();
            createDescriptorSets();
            updateDescriptorSets();
        }

      protected:
        template <typename CallableT>
        void forEachBuffer(CallableT /*callable*/) {
            assert(false && "Missing implementation for forEachBuffer()");
        }

        void bind(std::shared_ptr<environment::Device>& devicePointer,
                  std::shared_ptr<scaling::Base>&       scalingPointer) {
            forEachBuffer([&devicePointer, &scalingPointer](auto& buffer) {
                buffer.bind(devicePointer, scalingPointer);
            });
        };

        void allocateBuffers() {
            this->forEachBuffer([](auto& buffer) {
                buffer.allocateBuffers();
            });
        }

        void deallocateBuffers() {
            this->forEachBuffer([](auto& buffer) {
                buffer.allocateBuffers();
            });
        }

        void createDescriptorPool() {
            auto descriptorPoolSizes = getDescriptorPoolSizes();
            auto maxDescriptorSets   = getMaxDescriptorSets();

            this->descriptorPool = std::make_shared<vk::raii::DescriptorPool>(
                std::move(getDevice().getLogicalDevice().createDescriptorPool(
                    vk::DescriptorPoolCreateInfo()
                        .setFlags({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet})
                        .setPoolSizes(descriptorPoolSizes)
                        .setMaxSets(maxDescriptorSets))));
        }

        [[nodiscard]] std::vector<vk::DescriptorPoolSize> getDescriptorPoolSizes() {
            std::vector<vk::DescriptorPoolSize> poolSize{};

            uint64_t maxPoolSizes = 0;
            this->forEachBuffer([&maxPoolSizes](auto& /*buffer*/) {
                maxPoolSizes += 1;
            });
            poolSize.reserve(maxPoolSizes);

            this->forEachBuffer([&poolSize](auto& buffer) {
                poolSize.push_back(buffer.getDescriptorPoolSize());
            });

            return poolSize;
        };

        void createDescriptorSets() {
            auto raiiLayouts = getDescriptorSetLayouts();

            std::vector<vk::DescriptorSetLayout> layouts;
            layouts.reserve(raiiLayouts.size());

            std::transform(raiiLayouts.begin(),
                           raiiLayouts.end(),
                           layouts.begin(),
                           [](vk::raii::DescriptorSetLayout& val) {
                               return *val;
                           });

            this->descriptorSets = std::move(getDevice().getLogicalDevice().allocateDescriptorSets(
                vk::DescriptorSetAllocateInfo()
                    .setDescriptorPool(*getDescriptorPool())
                    .setDescriptorSetCount(getMaxDescriptorSets())
                    .setSetLayouts(layouts)));
        }

        [[nodiscard]] std::vector<vk::raii::DescriptorSetLayout> getDescriptorSetLayouts() {
            uint32_t maxSets = getMaxDescriptorSets();

            std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;
            descriptorSetLayouts.reserve(maxSets);

            for (uint32_t setIndex = 0; setIndex < maxSets; setIndex++) {
                auto descriptorSetLayoutBindings = getDescriptorSetLayoutBindings(setIndex);

                descriptorSetLayouts.push_back(
                    std::move(getDevice().getLogicalDevice().createDescriptorSetLayout(
                        vk::DescriptorSetLayoutCreateInfo()
                            .setFlags({})
                            .setBindings(descriptorSetLayoutBindings)
                            .setBindingCount(descriptorSetLayoutBindings.size()))));
            }

            return descriptorSetLayouts;
        }

        [[nodiscard]] uint32_t getMaxDescriptorSets() {
            uint32_t maxSets = 0;

            forEachBuffer([&maxSets](auto& buffer) {
                auto bufferMaxSets = buffer.getMaxDescriptorSets();
                if (bufferMaxSets > maxSets) {
                    maxSets = bufferMaxSets;
                }
            });

            return maxSets;
        }

        [[nodiscard]] virtual std::vector<vk::DescriptorSetLayoutBinding>
        getDescriptorSetLayoutBindings(uint32_t setIndex) {
            std::vector<vk::DescriptorSetLayoutBinding> setBindings{};

            uint64_t maxBindings = 0;
            forEachBuffer([&maxBindings](auto& /*buffer*/) {
                maxBindings += 1;
            });
            setBindings.reserve(maxBindings);

            forEachBuffer([&setBindings, setIndex](auto& buffer) {
                std::optional<vk::DescriptorSetLayoutBinding> optionalBinding =
                    buffer.getDescriptorSetLayoutBindings(setIndex);

                if (optionalBinding.has_value()) {
                    setBindings.push_back(optionalBinding.value());
                }
            });

            return setBindings;
        }

        void updateDescriptorSets() {
            uint32_t maxSets = getMaxDescriptorSets();
            for (uint32_t setIndex = 0; setIndex < maxSets; setIndex++) {
                auto descriptorSetWrites = getDescriptorSetWrites(setIndex);
            }
        }

        [[nodiscard]] std::vector<vk::WriteDescriptorSet>
        getDescriptorSetWrites(uint32_t setIndex) {
            return {};
        }

      protected:
        [[nodiscard]] environment::Device& getDevice() {
            assert(devicePointer);
            return *this->devicePointer;
        }

        [[nodiscard]] const environment::Device& getDevice() const {
            assert(devicePointer);
            return *this->devicePointer;
        }

        [[nodiscard]] vk::raii::DescriptorPool& getDescriptorPool() {
            assert(descriptorPool);
            return *this->descriptorPool;
        }

      private:
        std::shared_ptr<vk::raii::DescriptorPool> descriptorPool = {};
        std::vector<vk::raii::DescriptorSet>      descriptorSets = {};
        std::shared_ptr<environment::Device>      devicePointer  = {};
        std::shared_ptr<scaling::Base>            scalingPointer = {};
    };

    class Dynamic : public Base {
      public:
        Dynamic(std::vector<layout::Dynamic>&& inputBuffers_,
                std::vector<layout::Dynamic>&& deviceLocalBuffers_,
                std::vector<layout::Dynamic>&& outputBuffers_) {
            inputBuffers.reserve(inputBuffers_.size());
            for (auto& layout : inputBuffers_) {
                inputBuffers.emplace_back(layout);
            }
            deviceLocalBuffers.reserve(deviceLocalBuffers_.size());
            for (auto& layout : deviceLocalBuffers_) {
                deviceLocalBuffers.emplace_back(layout);
            }
            outputBuffers.reserve(outputBuffers_.size());
            for (auto& layout : outputBuffers_) {
                outputBuffers.emplace_back(layout);
            }
        }

      protected:
        template <typename CallableT>
        void forEachBuffer(CallableT callable) {
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
        std::vector<buffer::HostToDevice<layout::Dynamic>> inputBuffers       = {};
        std::vector<buffer::DeviceLocal<layout::Dynamic>>  deviceLocalBuffers = {};
        std::vector<buffer::DeviceToHost<layout::Dynamic>> outputBuffers      = {};
    };

    class Static : public Base {
      public:
        using Base::Base;
    };
} // namespace epseon::gpu::cpp::resources