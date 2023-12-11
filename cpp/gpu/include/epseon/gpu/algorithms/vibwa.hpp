#pragma once

#include "epseon/libepseon.hpp"
#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/task_configurator/algorithm_config.hpp"

#include "epseon/gpu/algorithms/algorithm.hpp"
#include "epseon/gpu/task_handle.hpp"
#include "vk_mem_alloc_handles.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace vma::raii {
    class Allocator : public vma::Allocator { // NOLINT: hicpp-special-member-functions
      public:
        using vma::Allocator::Allocator;

        ~Allocator() {
            this->destroy();
        }
    };
} // namespace vma::raii

namespace epseon::gpu::cpp {

    class Interrupted : public std::exception {};

    template <typename FP>
    class VibwaAlgorithm : public Algorithm<FP> {
        static_assert(std::is_floating_point<FP>::value, "FP must be an floating-point type.");

      public: /* Public constructors. */
        VibwaAlgorithm() :
            Algorithm<FP>() {}

      public: /* Public methods. */
        struct Buffer {};

        struct StorageBuffer : public Buffer {};

        struct UniformBuffer : public Buffer {};

        // Buffer which have to be transferred from host.
        // In reality this should contain to actual buffers:
        //  - one host visible
        //  - one device local
        struct InputBuffer : public Buffer {};

        // Buffer which should be only accessible from GPU.
        struct DeviceLocalBuffer : public Buffer {};

        // Buffer which should be writable by GPU and readable by CPU.
        struct OutputBuffer : public Buffer {};

        struct ShaderResources {
            std::shared_ptr<vma::raii::Allocator> allocator = {};

            std::vector<vk::Buffer>          stagingBuffers                 = {};
            std::vector<vma::Allocation>     stagingBuffersAllocations      = {};
            std::vector<vma::AllocationInfo> stagingBuffersAllocationsInfos = {};

            std::vector<vk::Buffer>      gpuOnlyStorageBuffers            = {};
            std::vector<vma::Allocation> gpuOnlyStorageBuffersAllocations = {};

            std::vector<vk::Buffer>          outputBuffers                 = {};
            std::vector<vma::Allocation>     outputBuffersAllocations      = {};
            std::vector<vma::AllocationInfo> outputBuffersAllocationsInfos = {};

          private:
            // Copy constructor
            explicit ShaderResources(std::shared_ptr<vma::raii::Allocator> allocator) :
                allocator(std::move(allocator)){};

          public:
            // Copy constructor
            ShaderResources() = default;

            // Copy constructor
            ShaderResources(const ShaderResources&) = delete;

            // Copy assignment operator
            ShaderResources& operator=(const ShaderResources&) = delete;

            ShaderResources(ShaderResources&& other) noexcept :
                allocator(std::move(other.allocator)),

                stagingBuffers(std::move(other.stagingBuffers)),
                stagingBuffersAllocations(std::move(other.stagingBuffersAllocations)),
                stagingBuffersAllocationsInfos(std::move(other.stagingBuffersAllocationsInfos)),

                gpuOnlyStorageBuffers(std::move(other.gpuOnlyStorageBuffers)),
                gpuOnlyStorageBuffersAllocations(std::move(other.gpuOnlyStorageBuffersAllocations)),
                outputBuffers(std::move(other.outputBuffers)),

                outputBuffersAllocations(std::move(other.outputBuffersAllocations)),
                outputBuffersAllocationsInfos(std::move(other.outputBuffersAllocationsInfos)) {}

            // Move assignment operator
            ShaderResources& operator=(ShaderResources&& other) noexcept {
                if (this != &other) {
                    // Move resources
                    allocator = std::move(other.allocator);

                    stagingBuffers            = std::move(other.stagingBuffers);
                    stagingBuffersAllocations = std::move(other.stagingBuffersAllocations);
                    stagingBuffersAllocationsInfos =
                        std::move(other.stagingBuffersAllocationsInfos);

                    gpuOnlyStorageBuffers = std::move(other.gpuOnlyStorageBuffers);
                    gpuOnlyStorageBuffersAllocations =
                        std::move(other.gpuOnlyStorageBuffersAllocations);

                    outputBuffers                 = std::move(other.outputBuffers);
                    outputBuffersAllocations      = std::move(other.outputBuffersAllocations);
                    outputBuffersAllocationsInfos = std::move(other.outputBuffersAllocationsInfos);
                }
                return *this;
            }

            ~ShaderResources() {
                destroy();
            }

            [[nodiscard]] uint32_t getGpuOnlyBufferCount() const {
                return this->gpuOnlyStorageBuffers.size();
            }

            [[nodiscard]] uint32_t getOutputBufferCount() const {
                return this->outputBuffers.size();
            }

          private:
            void destroy() {
                destroy(stagingBuffers, stagingBuffersAllocations);
                destroy(gpuOnlyStorageBuffers, gpuOnlyStorageBuffersAllocations);
                destroy(outputBuffers, outputBuffersAllocations);
            }

            template <typename BufferT, typename AllocationT>
            void destroy(std::vector<BufferT> buffers, std::vector<AllocationT> allocations) {
                LIB_EPSEON_ASSERT_TRUE(buffers.size() == allocations.size());

                for (uint32_t i = 0; i < buffers.size(); i++) {
                    allocator->destroyBuffer(buffers[i], allocations[i]);
                }
                buffers.clear();
                allocations.clear();
            }

          public:
            static ShaderResources create(std::shared_ptr<vma::raii::Allocator> allocator,
                                          const ShaderBuffersRequirements<FP>&  requirements) {
                ShaderResources resources{allocator};

                resources.stagingBuffers.reserve(requirements.stagingBuffersCount);
                resources.stagingBuffersAllocations.reserve(requirements.stagingBuffersCount);
                resources.stagingBuffersAllocationsInfos.reserve(requirements.stagingBuffersCount);

                resources.gpuOnlyStorageBuffers.reserve(requirements.gpuOnlyStorageBuffersCount);
                resources.gpuOnlyStorageBuffersAllocations.reserve(
                    requirements.gpuOnlyStorageBuffersCount);

                resources.outputBuffers.reserve(requirements.outputBuffersCount);
                resources.outputBuffersAllocations.reserve(requirements.outputBuffersCount);
                resources.outputBuffersAllocationsInfos.reserve(requirements.outputBuffersCount);

                /* Allocate staging buffers. */
                for (uint32_t i = 0; i < requirements.stagingBuffersCount; i++) {
                    vma::AllocationInfo info{};

                    auto [buffer, allocation] = allocator->createBuffer(
                        vk::BufferCreateInfo()
                            .setSize(requirements.getStagingBuffersSizeBytes())
                            .setUsage(vk::BufferUsageFlagBits::eTransferSrc),
                        vma::AllocationCreateInfo()
                            .setUsage(vma::MemoryUsage::eAuto)
                            .setFlags(vma::AllocationCreateFlagBits::eHostAccessSequentialWrite |
                                      vma::AllocationCreateFlagBits::eMapped),
                        &info);

                    resources.stagingBuffers.push_back(std::move(buffer));
                    resources.stagingBuffersAllocations.push_back(std::move(allocation));
                    resources.stagingBuffersAllocationsInfos.push_back(info);
                }
                /* Allocate GPU only buffers. */
                for (uint32_t i = 0; i < requirements.gpuOnlyStorageBuffersCount; i++) {
                    auto [buffer, allocation] = allocator->createBuffer(
                        vk::BufferCreateInfo()
                            .setSize(requirements.getGpuOnlyStorageBufferSizeBytes())
                            .setUsage(vk::BufferUsageFlagBits::eTransferDst |
                                      vk::BufferUsageFlagBits::eStorageBuffer),
                        vma::AllocationCreateInfo()
                            .setUsage(vma::MemoryUsage::eAuto)
                            .setFlags(vma::AllocationCreateFlagBits::eDedicatedMemory));

                    resources.gpuOnlyStorageBuffers.push_back(std::move(buffer));
                    resources.gpuOnlyStorageBuffersAllocations.push_back(std::move(allocation));
                }
                /* Allocate output (read back) buffers. */
                for (uint32_t i = 0; i < requirements.outputBuffersCount; i++) {
                    vma::AllocationInfo info{};

                    auto [buffer, allocation] = allocator->createBuffer(
                        vk::BufferCreateInfo()
                            .setSize(requirements.getOutputBufferSizeBytes())
                            .setUsage(vk::BufferUsageFlagBits::eStorageBuffer),
                        vma::AllocationCreateInfo()
                            .setUsage(vma::MemoryUsage::eAuto)
                            .setFlags(vma::AllocationCreateFlagBits::eHostAccessRandom |
                                      vma::AllocationCreateFlagBits::eMapped),
                        &info);

                    resources.outputBuffers.push_back(std::move(buffer));
                    resources.outputBuffersAllocations.push_back(std::move(allocation));
                    resources.outputBuffersAllocationsInfos.push_back(info);
                }

                return resources;
            }
        };

        struct ComputeBatchResources {
          private:
            uint32_t                                   shaderCount            = {};
            std::shared_ptr<vma::raii::Allocator>      allocator              = {};
            std::vector<ShaderResources>               shaderResources        = {};
            std::vector<vk::raii::DescriptorSetLayout> descriptorSetLayouts   = {};
            std::vector<vk::DescriptorSetLayout>       descriptorVkSetLayouts = {};
            std::shared_ptr<vk::raii::DescriptorPool>  descriptorPool         = {};
            std::vector<vk::raii::DescriptorSet>       descriptorSets         = {};
            std::shared_ptr<vk::raii::PipelineLayout>  pipelineLayout         = {};
            std::shared_ptr<vk::raii::Pipeline>        computePipeline        = {};

          private:
            explicit ComputeBatchResources(vma::raii::Allocator&& allocator) :
                allocator(std::make_shared<vma::raii::Allocator>(allocator)){};

          public:
            // Copy constructor
            ComputeBatchResources() = default;

            // Copy constructor
            ComputeBatchResources(const ComputeBatchResources&) = delete;

            // Copy assignment operator
            ComputeBatchResources& operator=(const ComputeBatchResources&) = delete;

            ComputeBatchResources(ComputeBatchResources&& other) noexcept :
                shaderCount(std::move(other.shaderCount)),
                allocator(std::move(other.allocator)),
                shaderResources(std::move(other.shaderResources)),
                descriptorSetLayouts(std::move(other.descriptorSetLayouts)),
                descriptorVkSetLayouts(std::move(other.descriptorVkSetLayouts)),
                descriptorPool(std::move(other.descriptorPool)) {}

            // Move assignment operator
            ComputeBatchResources& operator=(ComputeBatchResources&& other) noexcept {
                if (this != &other) {
                    // Move resources
                    shaderCount            = std::move(other.shaderCount);
                    allocator              = std::move(other.allocator);
                    shaderResources        = std::move(other.shaderResources);
                    descriptorSetLayouts   = std::move(other.descriptorSetLayouts);
                    descriptorVkSetLayouts = std::move(other.descriptorVkSetLayouts);
                    descriptorPool         = std::move(other.descriptorPool);
                }
                return *this;
            }

            ~ComputeBatchResources() = default;

            static ComputeBatchResources create(uint32_t                        vulkanApiVersion,
                                                const vk::raii::Instance&       instance,
                                                const vk::raii::PhysicalDevice& physicalDevice,
                                                const vk::raii::Device&         logicalDevice) {
                auto functions = vma::VulkanFunctions();

                functions.setVkGetInstanceProcAddr(instance.getDispatcher()->vkGetInstanceProcAddr)
                    .setVkGetDeviceProcAddr(instance.getDispatcher()->vkGetDeviceProcAddr);

                LIB_EPSEON_ASSERT_TRUE(functions.vkGetInstanceProcAddr != nullptr);
                LIB_EPSEON_ASSERT_TRUE(functions.vkGetDeviceProcAddr != nullptr);

                vma::Allocator allocator =
                    vma::createAllocator(vma::AllocatorCreateInfo()
                                             .setVulkanApiVersion(vulkanApiVersion)
                                             .setInstance(*instance)
                                             .setPhysicalDevice(*physicalDevice)
                                             .setDevice(*logicalDevice)
                                             .setPVulkanFunctions(&functions));
                ComputeBatchResources resources{
                    // Trust me I know what I am doing - vma::raii::Allocator is a transparent
                    // wrapper.
                    static_cast< // NOLINT: cppcoreguidelines-pro-type-static-cast-downcast
                        vma::raii::Allocator&&>(allocator)};

                return resources;
            }

            void allocateResources(const std::vector<ShaderBuffersRequirements<FP>>& requirements) {
                setShaderCount(requirements.size());
                shaderResources.reserve(getShaderCount());

                for (const auto& requirementStruct : requirements) {
                    shaderResources.emplace_back(
                        ShaderResources::create(allocator, requirementStruct));
                }
            }

            void setShaderCount(uint32_t shaderCount) {
                this->shaderCount = shaderCount;
            }

            [[nodiscard]] uint32_t getShaderCount() const {
                return this->shaderCount;
            }

            [[nodiscard]] uint32_t getDescriptorSetLayoutCount() const {
                return 1;
            }

            [[nodiscard]] uint32_t getDescriptorPoolCount() const {
                return 1;
            }

            [[nodiscard]] uint32_t getDescriptorSetCount() const {
                return 1;
            }

            [[nodiscard]] uint32_t getPerShaderGpuOnlyBufferCount() const {
                uint32_t expectedGpuOnlyBufferCount = 0;

                for (const ShaderResources& resource : shaderResources) {
                    uint32_t gpuOnlyStorageBuffersSize = resource.getGpuOnlyBufferCount();
#ifdef LIB_EPSEON_DEBUG
                    // All shaders must have same descriptor set layout. This should be already
                    // enforced by API, but better safe than sorry. Unfortunately this will cause
                    // std::terminate when std::jthread joins and hits throw below.
                    if (expectedGpuOnlyBufferCount != 0 &&
                        gpuOnlyStorageBuffersSize != expectedGpuOnlyBufferCount) {
                        throw std::runtime_error(
                            "All shader resources must have same GPU only buffer count.");
                    }
#endif
                    expectedGpuOnlyBufferCount = gpuOnlyStorageBuffersSize;
#ifdef LIB_EPSEON_RELEASE
                    break;
#endif
                }
                return expectedGpuOnlyBufferCount;
            }

            [[nodiscard]] vk::DescriptorType getGpuOnlyBufferDescriptorType() const {
                return vk::DescriptorType::eStorageBuffer;
            }

            [[nodiscard]] uint32_t getShaderOutputBufferCount() const {
                uint32_t expectedOutputBufferCount = 0;

                for (const ShaderResources& resource : shaderResources) {
                    uint32_t outputBuffersSize = resource.getOutputBufferCount();
#ifdef LIB_EPSEON_DEBUG
                    // All shaders must have same descriptor set layout. This should be already
                    // enforced by API, but better safe than sorry. Unfortunately this will cause
                    // std::terminate when std::jthread joins and hits throw below.
                    if (expectedOutputBufferCount != 0 &&
                        outputBuffersSize != expectedOutputBufferCount) {
                        throw std::runtime_error(
                            "All shader resources must have same GPU only buffer count.");
                    }
#endif
                    expectedOutputBufferCount = outputBuffersSize;
#ifdef LIB_EPSEON_RELEASE
                    break;
#endif
                }
                return expectedOutputBufferCount;
            }

            [[nodiscard]] vk::DescriptorType getOutputBufferDescriptorType() const {
                return vk::DescriptorType::eStorageBuffer;
            }

            [[nodiscard]] const std::vector<vk::DescriptorSetLayout>&
            getVkDescriptorSetLayouts() const {
                return this->descriptorVkSetLayouts;
            }

            void createDescriptorSets(const vk::raii::Device& logicalDevice) {
                createDescriptorSetLayouts(logicalDevice);
                createDescriptorPool(logicalDevice);

                this->descriptorSets = std::move(logicalDevice.allocateDescriptorSets(
                    vk::DescriptorSetAllocateInfo()
                        .setDescriptorPool(getDescriptorPool())
                        .setDescriptorSetCount(getDescriptorSetCount())
                        .setSetLayouts(getVkDescriptorSetLayouts())));
            }

            std::vector<vk::DescriptorSet> getDescriptorSets() {
                std::vector<vk::DescriptorSet> sets{};
                sets.reserve(descriptorSets.size());

                for (auto& descriptorSet : descriptorSets) {
                    sets.push_back(*descriptorSet);
                }
                return sets;
            }

          private:
            void createDescriptorSetLayouts(const vk::raii::Device& logicalDevice) {
                if (descriptorSetLayouts.empty()) {

                    uint32_t expectedGpuOnlyBufferCount = getPerShaderGpuOnlyBufferCount();
                    uint32_t outputBufferBindingOffset  = expectedGpuOnlyBufferCount;
                    uint32_t expectedOutputBufferCount  = getShaderOutputBufferCount();

                    LIB_EPSEON_ASSERT_TRUE(expectedGpuOnlyBufferCount > 0);
                    LIB_EPSEON_ASSERT_TRUE(expectedOutputBufferCount > 0);

                    std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings{};
                    // We can specify upfront our needs at runtime, but we can't deduce them at
                    // compile time. Reserving memory for exact number of elements before
                    // push_back() will allow us to avoid resource reallocation.
                    descriptorSetLayoutBindings.reserve(expectedGpuOnlyBufferCount +
                                                        expectedOutputBufferCount);

                    for (uint32_t binding = 0; binding < expectedGpuOnlyBufferCount; binding++) {
                        descriptorSetLayoutBindings.push_back(
                            vk::DescriptorSetLayoutBinding()
                                .setBinding(binding)
                                .setDescriptorCount(getShaderCount())
                                .setDescriptorType(getGpuOnlyBufferDescriptorType())
                                .setStageFlags({vk::ShaderStageFlagBits::eCompute}));
                    }
                    LIB_EPSEON_ASSERT_TRUE(!descriptorSetLayoutBindings.empty());

                    for (uint32_t binding = outputBufferBindingOffset;
                         binding < (expectedOutputBufferCount + outputBufferBindingOffset);
                         binding++) {
                        descriptorSetLayoutBindings.push_back(
                            vk::DescriptorSetLayoutBinding()
                                .setBinding(binding)
                                .setDescriptorCount(getShaderCount())
                                .setDescriptorType(getOutputBufferDescriptorType())
                                .setStageFlags({vk::ShaderStageFlagBits::eCompute}));
                    }
                    LIB_EPSEON_ASSERT_TRUE(!descriptorSetLayoutBindings.empty());

                    if (!descriptorSetLayoutBindings.empty()) {
                        // We need exactly one for now, so avoid allocation of space for multiple
                        // elements.
                        descriptorSetLayouts.reserve(getDescriptorSetLayoutCount());
                        descriptorSetLayouts.push_back(
                            std::move(logicalDevice.createDescriptorSetLayout(
                                vk::DescriptorSetLayoutCreateInfo().setBindings(
                                    descriptorSetLayoutBindings))));

                        descriptorVkSetLayouts.reserve(getDescriptorSetLayoutCount());
                        for (auto& descriptorSetLayout : descriptorSetLayouts) {
                            descriptorVkSetLayouts.push_back(*descriptorSetLayout);
                        }
                    }
                }
            }

            void createDescriptorPool(const vk::raii::Device& logicalDevice) {
                LIB_EPSEON_ASSERT_FALSE(descriptorPool);

                std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;

                for (uint32_t binding = 0; binding < getPerShaderGpuOnlyBufferCount(); binding++) {
                    descriptorPoolSizes.push_back(vk::DescriptorPoolSize()
                                                      .setDescriptorCount(getShaderCount())
                                                      .setType(vk::DescriptorType::eStorageBuffer));
                }
                for (uint32_t binding = 0; binding < getShaderOutputBufferCount(); binding++) {
                    descriptorPoolSizes.push_back(vk::DescriptorPoolSize()
                                                      .setDescriptorCount(getShaderCount())
                                                      .setType(vk::DescriptorType::eStorageBuffer));
                }
                if (!descriptorPoolSizes.empty()) {
                    descriptorPool = std::make_shared<vk::raii::DescriptorPool>(
                        std::move(logicalDevice.createDescriptorPool(
                            vk::DescriptorPoolCreateInfo()
                                .setFlags({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet})
                                .setPoolSizes(descriptorPoolSizes)
                                .setMaxSets(getDescriptorSetCount()))));
                }
            }

            [[nodiscard]] vk::DescriptorPool getDescriptorPool() {
                LIB_EPSEON_ASSERT_TRUE(this->descriptorPool);
                return *(*this->descriptorPool);
            }

            // We have to keep std::vector<vk::DescriptorBufferInfo> alive until
            // vk::WriteDescriptorSet needs it. Simplest way is to simply store them
            // together.
            struct WriteDescriptorSet {
                vk::WriteDescriptorSet                writeDescriptorSet = {};
                std::vector<vk::DescriptorBufferInfo> bufferInfo         = {};

                void fillBufferInfoVector(
                    const std::vector<ShaderResources>&               shaderResources,
                    std::function<vk::Buffer(const ShaderResources&)> bufferGetter) {
                    uint32_t resourceCount = shaderResources.size();
                    bufferInfo.reserve(resourceCount);
                    bufferInfo.resize(resourceCount);

                    for (uint32_t arrayIndex = 0; arrayIndex < resourceCount; arrayIndex++) {
                        const ShaderResources& resource = shaderResources[arrayIndex];
                        LIB_EPSEON_ASSERT_TRUE(arrayIndex < bufferInfo.size());

                        bufferInfo[arrayIndex]
                            .setBuffer(bufferGetter(resource))
                            .setOffset(0)
                            .setRange(vk::WholeSize);
                    }
                }
            };

          private:
            std::vector<WriteDescriptorSet>
            getDescriptorSetWrites(vk::raii::DescriptorSet& descriptorSet) {
                std::vector<WriteDescriptorSet> descriptorSetWrites;

                uint32_t expectedGpuOnlyBufferCount = getPerShaderGpuOnlyBufferCount();
                uint32_t outputBufferBindingOffset  = expectedGpuOnlyBufferCount;
                uint32_t expectedOutputBufferCount  = getShaderOutputBufferCount();

                descriptorSetWrites.reserve(expectedGpuOnlyBufferCount + expectedOutputBufferCount);
                descriptorSetWrites.resize(expectedGpuOnlyBufferCount + expectedOutputBufferCount);

                for (uint32_t binding = 0; binding < expectedGpuOnlyBufferCount; binding++) {
                    WriteDescriptorSet& write       = descriptorSetWrites[binding];
                    uint32_t            bufferIndex = binding;
                    write.fillBufferInfoVector(
                        shaderResources, [bufferIndex](const ShaderResources& resource) {
                            LIB_EPSEON_ASSERT_TRUE(bufferIndex <
                                                   resource.gpuOnlyStorageBuffers.size());
                            return resource.gpuOnlyStorageBuffers[bufferIndex];
                        });
                    write.writeDescriptorSet.setDstSet(*descriptorSet)
                        .setDstBinding(binding)
                        .setBufferInfo(write.bufferInfo)
                        .setDescriptorCount(getShaderCount())
                        .setDescriptorType(getGpuOnlyBufferDescriptorType());
                }

                for (uint32_t binding = outputBufferBindingOffset;
                     binding < (expectedOutputBufferCount + outputBufferBindingOffset);
                     binding++) {
                    WriteDescriptorSet& write       = descriptorSetWrites[binding];
                    uint32_t            bufferIndex = binding - outputBufferBindingOffset;
                    write.fillBufferInfoVector(
                        shaderResources, [bufferIndex](const ShaderResources& resource) {
                            LIB_EPSEON_ASSERT_TRUE(bufferIndex < resource.outputBuffers.size());
                            return resource.outputBuffers[bufferIndex];
                        });
                    write.writeDescriptorSet.setDstSet(*descriptorSet)
                        .setDstBinding(binding)
                        .setBufferInfo(write.bufferInfo)
                        .setDescriptorCount(getShaderCount())
                        .setDescriptorType(getOutputBufferDescriptorType());
                }

                return descriptorSetWrites;
            }

          public:
            void updateDescriptorSets(const vk::raii::Device& logicalDevice) {
                for (vk::raii::DescriptorSet& descriptorSet : this->descriptorSets) {
                    auto writes = getDescriptorSetWrites(descriptorSet);
                    for (WriteDescriptorSet& write : writes) {
                        logicalDevice.updateDescriptorSets(write.writeDescriptorSet, {});
                    }
                }
            }

            std::vector<uint32_t> getShaderCode() {
                std::string filename = "cpp/gpu/shaders/main.comp.spv";

                std::ifstream file(filename, std::ios::ate | std::ios::binary);

                if (!file.is_open()) {
                    throw std::runtime_error("Failed to open file: " + filename);
                }

                uint32_t              fileSize = static_cast<size_t>(file.tellg());
                std::vector<uint32_t> buffer(fileSize);

                file.seekg(0);
                file.read(
                    reinterpret_cast<char*>( // NOLINT: cppcoreguidelines-pro-type-reinterpret-cast
                        buffer.data()),
                    fileSize);

                file.close();
                return buffer;
            }

            void createComputePipeline(const vk::raii::Device& logicalDevice) {

                auto shaderCode = getShaderCode();
                auto shaderModule =
                    logicalDevice.createShaderModule(vk::ShaderModuleCreateInfo()
                                                         .setPCode(shaderCode.data())
                                                         .setCodeSize(shaderCode.size()));

                this->pipelineLayout = std::make_shared<vk::raii::PipelineLayout>(std::move(
                    logicalDevice.createPipelineLayout(vk::PipelineLayoutCreateInfo().setSetLayouts(
                        getVkDescriptorSetLayouts()))));

                this->computePipeline = std::make_shared<vk::raii::Pipeline>(
                    std::move(logicalDevice.createComputePipeline(
                        nullptr,
                        vk::ComputePipelineCreateInfo()
                            .setStage(vk::PipelineShaderStageCreateInfo()
                                          .setStage(vk::ShaderStageFlagBits::eCompute)
                                          .setModule(*shaderModule)
                                          .setPName("main"))
                            .setLayout(*getPipelineLayout()))));
            }

            vk::raii::PipelineLayout& getPipelineLayout() {
                return *this->pipelineLayout;
            }

            vk::raii::Pipeline& getComputePipeline() {
                return *this->computePipeline;
            }

            void recordBufferTransfers(vk::raii::CommandBuffer& commandBuffer) {
                for (ShaderResources& resource : this->shaderResources) {}
            }

            void validateResult() {
                for (ShaderResources& resource : this->shaderResources) {
                    auto&            info = resource.outputBuffersAllocationsInfos.front();
                    std::span<float> data{static_cast<float*>(info.pMappedData),
                                          static_cast<float*>(info.pMappedData) + 40};
                    for (uint32_t index = 0; index < data.size(); index++) {
                        auto value = data[index];
                    }
                }
            }
        };

        virtual void run(const std::stop_token& stop_token, TaskHandle<FP>* handle) {
            if (stop_token.stop_requested()) {
                return;
            }

            const auto& physicalDevice = handle->getDeviceInterface().getPhysicalDevice();
            auto [logicalDevice, queueFamilyIndex] = createLogicalDevice(physicalDevice);
            const auto& compute_context_state =
                handle->getDeviceInterface().getComputeContextState();

            auto configurator = handle->getTaskConfigurator();

            if (stop_token.stop_requested()) {
                return;
            }

            ComputeBatchResources resources = ComputeBatchResources::create(
                handle->getDeviceInterface().getComputeContextState().getVulkanApiVersion(),
                compute_context_state.getVkInstance(),
                physicalDevice,
                logicalDevice);
            auto requirements = configurator.getShaderBufferRequirements();
            resources.allocateResources(requirements);
            resources.createDescriptorSets(logicalDevice);
            resources.updateDescriptorSets(logicalDevice);
            resources.createComputePipeline(logicalDevice);

            if (stop_token.stop_requested()) {
                return;
            }

            CommandBufferResources commandBufferResources =
                CommandBufferResources::create(logicalDevice, queueFamilyIndex);

            vk::raii::CommandBuffer& commandBuffer = commandBufferResources.getCommandBuffer(0);

            auto inputBarrier = vk::MemoryBarrier(vk::AccessFlagBits::eHostWrite,
                                                  vk::AccessFlagBits::eTransferRead);
            auto outputBarrier =
                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead);

            commandBuffer.begin({});
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                       *resources.getComputePipeline());
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                             *resources.getPipelineLayout(),
                                             0,
                                             resources.getDescriptorSets(),
                                             nullptr);

            // Ensure that all host writes happened before transfer to GPU.
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                          vk::PipelineStageFlagBits::eTransfer,
                                          {},
                                          inputBarrier,
                                          {},
                                          {});

            resources.recordBufferTransfers(commandBuffer);

            commandBuffer.dispatch(resources.getShaderCount(), 1, 1);
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                          vk::PipelineStageFlagBits::eAllCommands,
                                          {},
                                          outputBarrier,
                                          {},
                                          {});
            commandBuffer.end();

            auto fence = logicalDevice.createFence({});

            commandBufferResources.getQueue().submit(
                vk::SubmitInfo().setCommandBufferCount(1).setCommandBuffers(*commandBuffer),
                *fence);

            const uint64_t fiveSeconds = 5000000000UL /* nanoseconds */;

            auto result = logicalDevice.waitForFences({*fence}, true, fiveSeconds);
            resources.validateResult();
            std::cout << "Thread finished" << std::endl;
        }

        uint32_t selectQueueFamilyIndex(const vk::raii::PhysicalDevice& physicalDevice) {
            uint32_t queueFamilyIndex = -1;

            for (auto i = 0; const auto& queue : physicalDevice.getQueueFamilyProperties()) {
                if ((queue.queueFlags & vk::QueueFlagBits::eCompute) &&
                    (queue.queueFlags & vk::QueueFlagBits::eTransfer)) {
                    queueFamilyIndex = i;
                    break;
                }
                i++;
            }
            LIB_EPSEON_ASSERT_TRUE(queueFamilyIndex != -1);
            return queueFamilyIndex;
        }

        std::pair<vk::raii::Device, uint32_t>
        createLogicalDevice(const vk::raii::PhysicalDevice& physicalDevice) {
            // Exclusive transfer Queue in some GPUs, we may use it in
            // future.
            uint32_t             queueIndex      = selectQueueFamilyIndex(physicalDevice);
            std::array<float, 1> queuePriorities = {1.0F};
            std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{
                vk::DeviceQueueCreateInfo()
                    .setQueueFamilyIndex(queueIndex)
                    .setQueuePriorities(queuePriorities)};

            std::vector<const char*> requiredDeviceExtensions{};

            auto featuresChain =
                physicalDevice.getFeatures2<vk::PhysicalDeviceFeatures2,
                                            vk::PhysicalDeviceDescriptorIndexingFeatures>();

            auto deviceFeatures = featuresChain.get<vk::PhysicalDeviceFeatures2>();
            if (!static_cast<bool>(
                    deviceFeatures.features.shaderStorageBufferArrayDynamicIndexing)) {
                std::cout
                    << "Warning: shaderStorageBufferArrayDynamicIndexing feature not available."
                    << std::endl;
            }

            auto descriptorIndexingFeatures =
                featuresChain.get<vk::PhysicalDeviceDescriptorIndexingFeatures>();

            if (!static_cast<bool>(
                    descriptorIndexingFeatures.shaderStorageBufferArrayNonUniformIndexing)) {
                std::cout
                    << "Warning: shaderStorageBufferArrayNonUniformIndexing feature not available."
                    << std::endl;
            } else {
                requiredDeviceExtensions.push_back("VK_EXT_descriptor_indexing");
            }
            LIB_EPSEON_ASSERT_TRUE(deviceFeatures.pNext != nullptr);
            deviceFeatures.setPNext(&descriptorIndexingFeatures);

            auto logicalDevice = physicalDevice.createDevice(
                vk::DeviceCreateInfo()
                    .setPNext(&deviceFeatures)
                    .setQueueCreateInfos(queueCreateInfos)
                    .setPEnabledExtensionNames(requiredDeviceExtensions));

            return {std::move(logicalDevice), queueIndex};
        }

        class CommandBufferResources {
            vk::raii::CommandPool                commandPool;
            std::vector<vk::raii::CommandBuffer> commandBuffers;
            vk::raii::Queue                      commandQueue;

          public:
            // Default constructor.
            CommandBufferResources() = delete;

            // Copy constructor.
            CommandBufferResources(const CommandBufferResources&) = delete;

            // Copy assignment operator.
            CommandBufferResources& operator=(const CommandBufferResources&) = delete;

            // Move constructor.
            CommandBufferResources(CommandBufferResources&& other) noexcept = delete;

            // Move assignment operator.
            CommandBufferResources& operator=(CommandBufferResources&& other) noexcept = delete;

            ~CommandBufferResources() = default;

            [[nodiscard]] static CommandBufferResources create(vk::raii::Device& logicalDevice,
                                                               uint32_t          queueFamilyIndex) {
                vk::raii::CommandPool commandPool = logicalDevice.createCommandPool(
                    vk::CommandPoolCreateInfo()
                        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                        .setQueueFamilyIndex(queueFamilyIndex));
                std::vector<vk::raii::CommandBuffer> commandBuffers =
                    logicalDevice.allocateCommandBuffers(
                        vk::CommandBufferAllocateInfo()
                            .setCommandBufferCount(1)
                            .setCommandPool(*commandPool)
                            .setLevel(vk::CommandBufferLevel::ePrimary));
                vk::raii::Queue commandQueue = logicalDevice.getQueue(queueFamilyIndex, 0);

                return CommandBufferResources(
                    std::move(commandPool), std::move(commandBuffers), std::move(commandQueue));
            }

          private:
            explicit CommandBufferResources(vk::raii::CommandPool                commandPool_,
                                            std::vector<vk::raii::CommandBuffer> commandBuffers_,
                                            vk::raii::Queue                      commandQueue_) :
                commandPool(std::move(commandPool_)),
                commandBuffers(std::move(commandBuffers_)),
                commandQueue(std::move(commandQueue_)) {}

          public:
            [[nodiscard]] const vk::raii::CommandBuffer& getCommandBuffer(uint32_t index) const {
                return this->commandBuffers[index];
            }

            [[nodiscard]] vk::raii::CommandBuffer& getCommandBuffer(uint32_t index) {
                return this->commandBuffers[index];
            }

            [[nodiscard]] const vk::raii::Queue& getQueue() const {
                return this->commandQueue;
            }

            [[nodiscard]] vk::raii::Queue& getQueue() {
                return this->commandQueue;
            }
        };
    };
} // namespace epseon::gpu::cpp

#include "epseon/gpu/compute/shader.hpp"
