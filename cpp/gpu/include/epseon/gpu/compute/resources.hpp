#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/allocation.hpp"
#include "epseon/gpu/compute/buffer.hpp"
#include "epseon/gpu/compute/layout.hpp"
#include "epseon/gpu/compute/scaling.hpp"

#include "vk_mem_alloc.h"
#include "vk_mem_alloc_handles.hpp"
#include "vk_mem_alloc_structs.hpp"
#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace epseon::gpu::cpp::resources {

    template <typename T>
    concept Concept = requires(T t, std::shared_ptr<vma::Allocator>& allocatorT) {
        { t.iterateBuffers(allocatorT) } -> std::same_as<void>;
    };

    class Base {
      public:
        Base()                      = default;
        Base(const Base& other)     = delete;
        Base(Base&& other) noexcept = default;

        virtual ~Base() = default;

        Base& operator=(const Base& other)     = delete;
        Base& operator=(Base&& other) noexcept = default;

        template <typename CallableT>
        void iterateBuffers(CallableT /*callable*/) {
            throw std::runtime_error("iterateBuffers must be implemented.");
        }

        void bindDeviceAllocator(std::shared_ptr<vma::Allocator>& allocator) {
            iterateBuffers([&allocator](auto& buffer) {
                buffer.bindDeviceAllocator(allocator);
            });
        };

      private:
        [[nodiscard]] std::vector<vk::DescriptorPoolSize> getDescriptorPoolSize() {
            std::vector<vk::DescriptorPoolSize> poolSize{};

            uint64_t maxPoolSizes = 0;
            this->iterateBuffers([&maxPoolSizes](auto& /*buffer*/) {
                maxPoolSizes += 1;
            });
            poolSize.reserve(maxPoolSizes);

            this->iterateBuffers([&poolSize](auto& buffer) {
                poolSize.push_back(buffer.getDescriptorPoolSize());
            });

            return poolSize;
        };

        [[nodiscard]] std::vector<vk::raii::DescriptorSetLayout>
        getDescriptorSetLayouts(vk::raii::Device& logicalDevice) {
            uint32_t maxSets = getMaxDescriptorSets();

            std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts;
            descriptorSetLayouts.reserve(maxSets);

            for (uint32_t setIndex = 0; setIndex < maxSets; setIndex++) {
                auto descriptorSetLayoutBindings = getDescriptorSetLayoutBindings(setIndex);

                descriptorSetLayouts.push_back(std::move(logicalDevice.createDescriptorSetLayout(
                    vk::DescriptorSetLayoutCreateInfo()
                        .setFlags({})
                        .setBindings(descriptorSetLayoutBindings)
                        .setBindingCount(descriptorSetLayoutBindings.size()))));
            }

            return descriptorSetLayouts;
        }

        [[nodiscard]] uint32_t getMaxDescriptorSets() {
            uint32_t maxSets = 0;

            iterateBuffers([&maxSets](auto& buffer) {
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
            iterateBuffers([&maxBindings](auto& /*buffer*/) {
                maxBindings += 1;
            });
            setBindings.reserve(maxBindings);

            iterateBuffers([&setBindings, setIndex](auto& buffer) {
                std::optional<vk::DescriptorSetLayoutBinding> optionalBinding =
                    buffer.getDescriptorSetLayoutBindings(setIndex);

                if (optionalBinding.has_value()) {
                    setBindings.push_back(optionalBinding.value());
                }
            });

            return setBindings;
        }

      private:
        std::unique_ptr<vk::raii::DescriptorPool> descriptorPool = {};
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
        std::vector<buffer::HostToDevice<layout::Dynamic>> inputBuffers       = {};
        std::vector<buffer::DeviceLocal<layout::Dynamic>>  deviceLocalBuffers = {};
        std::vector<buffer::DeviceToHost<layout::Dynamic>> outputBuffers      = {};
    };

    class Static : public Base {
      public:
        using Base::Base;
    };
} // namespace epseon::gpu::cpp::resources