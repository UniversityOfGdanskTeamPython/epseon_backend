#pragma once

#include "epseon/gpu/compute/spirv.hpp"
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
#include <algorithm>
#include <cassert>
#include <iterator>
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

    template <class Derived>
    class Base {
      public:
        Base()                      = default;
        Base(const Base& other)     = delete;
        Base(Base&& other) noexcept = default;

        virtual ~Base() = default;

        Base& operator=(const Base& other)     = delete;
        Base& operator=(Base&& other) noexcept = default;

        void prepare(SPIRV&                                bytecode,
                     std::shared_ptr<environment::Device>& devicePointer,
                     std::shared_ptr<scaling::Base>&       scalingPointer) {
            bind(devicePointer, scalingPointer);
            allocateBuffers();
            createDescriptorPool();
            createDescriptorSets();
            updateDescriptorSets();
            createPipeline(bytecode);
            createCommandPool();
            createCommandBuffer();
            recordCommandBuffers();
        }

        template <typename CallableT>
        void forEachBuffer(CallableT /*callable*/) {
            assert(false && "Missing implementation for forEachBuffer()");
        }

      protected:
        void bind(std::shared_ptr<environment::Device>& devicePointer,
                  std::shared_ptr<scaling::Base>&       scalingPointer) {
            this->devicePointer  = devicePointer;
            this->scalingPointer = scalingPointer;
            static_cast<Derived*>(this)->forEachBuffer(
                [&devicePointer, &scalingPointer](auto& buffer) {
                    buffer.bind(devicePointer, scalingPointer);
                });
        };

        void allocateBuffers() {
            static_cast<Derived*>(this)->forEachBuffer([](auto& buffer) {
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
            static_cast<Derived*>(this)->forEachBuffer([&maxPoolSizes](auto& /*buffer*/) {
                maxPoolSizes += 1;
            });
            poolSize.reserve(maxPoolSizes);

            static_cast<Derived*>(this)->forEachBuffer([&poolSize](auto& buffer) {
                poolSize.push_back(buffer.getDescriptorPoolSize());
            });

            return poolSize;
        };

        void createDescriptorSets() {
            createDescriptorSetLayouts();
            std::vector<vk::DescriptorSetLayout> layouts = getDescriptorSetLayouts();

            this->descriptorSets = std::move(getDevice().getLogicalDevice().allocateDescriptorSets(
                vk::DescriptorSetAllocateInfo()
                    .setDescriptorPool(*getDescriptorPool())
                    .setDescriptorSetCount(getMaxDescriptorSets())
                    .setSetLayouts(layouts)));
        }

        void createDescriptorSetLayouts() {
            uint32_t maxSets = getMaxDescriptorSets();
            raiiDescriptorSetLayouts.reserve(maxSets);

            for (uint32_t setIndex = 0; setIndex < maxSets; setIndex++) {
                auto descriptorSetLayoutBindings = getDescriptorSetLayoutBindings(setIndex);

                raiiDescriptorSetLayouts.push_back(
                    std::move(getDevice().getLogicalDevice().createDescriptorSetLayout(
                        vk::DescriptorSetLayoutCreateInfo()
                            .setFlags({})
                            .setBindings(descriptorSetLayoutBindings)
                            .setBindingCount(descriptorSetLayoutBindings.size()))));
            }
        }

        [[nodiscard]] std::vector<vk::DescriptorSetLayout> getDescriptorSetLayouts() {
            std::vector<vk::DescriptorSetLayout> layouts;
            layouts.reserve(raiiDescriptorSetLayouts.size());

            std::transform(raiiDescriptorSetLayouts.begin(),
                           raiiDescriptorSetLayouts.end(),
                           std::back_inserter(layouts),
                           [](vk::raii::DescriptorSetLayout& val) {
                               return *val;
                           });

            return layouts;
        }

        [[nodiscard]] uint32_t getMaxDescriptorSets() {
            uint32_t maxSets = 0;

            static_cast<Derived*>(this)->forEachBuffer([&maxSets](auto& buffer) {
                auto bufferMaxSets = buffer.getMaxDescriptorSets();
                if (bufferMaxSets > maxSets) {
                    maxSets = bufferMaxSets;
                }
            });

            return (maxSets + 1);
        }

        [[nodiscard]] std::vector<vk::DescriptorSetLayoutBinding>
        getDescriptorSetLayoutBindings(uint32_t setIndex) {
            std::vector<vk::DescriptorSetLayoutBinding> setBindings{};

            uint64_t maxBindings = 0;
            static_cast<Derived*>(this)->forEachBuffer([&maxBindings](auto& /*buffer*/) {
                maxBindings += 1;
            });
            setBindings.reserve(maxBindings);

            static_cast<Derived*>(this)->forEachBuffer([&setBindings, setIndex](auto& buffer) {
                std::optional<vk::DescriptorSetLayoutBinding> optionalBinding =
                    buffer.getDescriptorSetLayoutBindings(setIndex);

                if (optionalBinding.has_value()) {
                    setBindings.push_back(optionalBinding.value());
                }
            });

            return setBindings;
        }

        void updateDescriptorSets() {
            vk::raii::Device& logicalDevice = getDevice().getLogicalDevice();

            static_cast<Derived*>(this)->forEachBuffer([&logicalDevice, this](auto& buffer) {
                auto     layout       = buffer.getLayout();
                uint32_t setIndex     = layout.getSet();
                uint32_t bindingIndex = layout.getSet();

                vk::raii::DescriptorSet& descriptorSet = this->descriptorSets[setIndex];

                DescriptorSetWrite writeInfo = buffer.getDescriptorSetWrite(descriptorSet);
                logicalDevice.updateDescriptorSets(writeInfo.writeInfo, {});
            });
        }

        void createPipeline(SPIRV& bytecode) {
            vk::raii::Device&                    logicalDevice = getDevice().getLogicalDevice();
            std::vector<vk::DescriptorSetLayout> layouts       = getDescriptorSetLayouts();

            computeShaderModule =
                std::make_unique<vk::raii::ShaderModule>(std::move(logicalDevice.createShaderModule(
                    vk::ShaderModuleCreateInfo().setCode(bytecode.getCode()))));

            computePipelineLayout = std::make_unique<vk::raii::PipelineLayout>(
                std::move(logicalDevice.createPipelineLayout(
                    vk::PipelineLayoutCreateInfo().setSetLayoutCount(1).setSetLayouts(layouts))));

            computePipeline =
                std::make_unique<vk::raii::Pipeline>(std::move(logicalDevice.createComputePipeline(
                    nullptr,
                    vk::ComputePipelineCreateInfo()
                        .setStage(vk::PipelineShaderStageCreateInfo()
                                      .setStage(vk::ShaderStageFlagBits::eCompute)
                                      .setModule(**computeShaderModule)
                                      .setPName("main"))
                        .setLayout(**computePipelineLayout))));
        }

        void createCommandPool() {
            commandPool = std::make_unique<vk::raii::CommandPool>(
                std::move(getDevice().getLogicalDevice().createCommandPool(
                    vk::CommandPoolCreateInfo()
                        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                        .setQueueFamilyIndex(getDevice().getQueueFamilyIndex()))));
        }

        void createCommandBuffer() {
            commandBuffers = getDevice().getLogicalDevice().allocateCommandBuffers(
                vk::CommandBufferAllocateInfo()
                    .setCommandPool(**commandPool)
                    .setLevel(vk::CommandBufferLevel::ePrimary)
                    .setCommandBufferCount(1));
        }

        void recordCommandBuffers() {
            auto& commandBuffer = commandBuffers.front();
            commandBuffer.begin({});
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, **computePipeline);

            std::vector<vk::DescriptorSet> descriptorSetsArray{};
            descriptorSetsArray.reserve(descriptorSets.size());

            std::transform(descriptorSets.begin(),
                           descriptorSets.end(),
                           std::back_inserter(descriptorSetsArray),
                           [](vk::raii::DescriptorSet& val) {
                               return *val;
                           });

            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             **computePipelineLayout,
                                             0,
                                             descriptorSetsArray,
                                             nullptr);

            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                          vk::PipelineStageFlagBits::eTransfer,
                                          {},
                                          vk::MemoryBarrier(vk::AccessFlagBits::eHostWrite,
                                                            vk::AccessFlagBits::eTransferRead),
                                          {},
                                          {});

            static_cast<Derived*>(this)->forEachBuffer([&commandBuffer](auto& buffer) {
                buffer.recordHostToDeviceTransfers(commandBuffer);
            });

            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                          vk::PipelineStageFlagBits::eComputeShader,
                                          {},
                                          vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite,
                                                            vk::AccessFlagBits::eShaderRead),
                                          {},
                                          {});

            commandBuffer.dispatch(getScaling().getBatchSize(), 1, 1);

            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                          vk::PipelineStageFlagBits::eTransfer,
                                          {},
                                          vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite,
                                                            vk::AccessFlagBits::eTransferRead),
                                          {},
                                          {});

            static_cast<Derived*>(this)->forEachBuffer([&commandBuffer](auto& buffer) {
                buffer.recordDeviceToHostTransfers(commandBuffer);
            });

            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                          vk::PipelineStageFlagBits::eAllCommands,
                                          {},
                                          vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite,
                                                            vk::AccessFlagBits::eHostRead),
                                          {},
                                          {});

            commandBuffer.end();
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

        [[nodiscard]] const scaling::Base& getScaling() const {
            assert(scalingPointer);
            return *this->scalingPointer;
        }

        [[nodiscard]] vk::raii::DescriptorPool& getDescriptorPool() {
            assert(descriptorPool);
            return *this->descriptorPool;
        }

      private:
        std::shared_ptr<vk::raii::DescriptorPool>  descriptorPool           = {};
        std::vector<vk::raii::DescriptorSet>       descriptorSets           = {};
        std::shared_ptr<environment::Device>       devicePointer            = {};
        std::shared_ptr<scaling::Base>             scalingPointer           = {};
        std::vector<vk::raii::DescriptorSetLayout> raiiDescriptorSetLayouts = {};
        std::unique_ptr<vk::raii::CommandPool>     commandPool              = {};
        std::vector<vk::raii::CommandBuffer>       commandBuffers           = {};
        std::unique_ptr<vk::raii::PipelineLayout>  computePipelineLayout    = {};
        std::unique_ptr<vk::raii::Pipeline>        computePipeline          = {};
        std::unique_ptr<vk::raii::ShaderModule>    computeShaderModule      = {};
    };

    class Dynamic : public Base<Dynamic> {

        using inputBufferT     = buffer::HostToDevice<layout::Dynamic>;
        using temporaryBufferT = buffer::DeviceLocal<layout::Dynamic>;
        using outputBufferT    = buffer::DeviceToHost<layout::Dynamic>;

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

        Dynamic()                         = default;
        Dynamic(const Dynamic& other)     = delete;
        Dynamic(Dynamic&& other) noexcept = default;

        ~Dynamic() override = default;

        Dynamic& operator=(const Dynamic& other)     = delete;
        Dynamic& operator=(Dynamic&& other) noexcept = default;

        inputBufferT& getInput(uint32_t index) {
            return this->inputBuffers[index];
        }

        outputBufferT& getOutput(uint32_t index) {
            return this->outputBuffers[index];
        }

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
        std::vector<inputBufferT>     inputBuffers       = {};
        std::vector<temporaryBufferT> deviceLocalBuffers = {};
        std::vector<outputBufferT>    outputBuffers      = {};
    };

    template <typename Derived>
    class Static : public Base<Derived> {
      public:
        Static()                        = default;
        Static(const Static& other)     = delete;
        Static(Static&& other) noexcept = default;

        ~Static() override = default;

        Static& operator=(const Static& other)     = delete;
        Static& operator=(Static&& other) noexcept = default;
    };
} // namespace epseon::gpu::cpp::resources
