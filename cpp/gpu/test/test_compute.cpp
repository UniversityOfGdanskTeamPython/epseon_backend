#include "epseon/gpu/compute/environment.hpp"
#include "epseon/gpu/compute/layout.hpp"
#include "epseon/gpu/compute/predecl.hpp"
#include "epseon/gpu/compute/resources.hpp"
#include "epseon/gpu/compute/scaling.hpp"
#include "epseon/gpu/compute/shader.hpp"
#include "epseon/gpu/compute/spirv.hpp"
#include <algorithm>
#include <csignal>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace epseon::gpu::cpp {

    class StaticThereAndBackResource : public resources::Static<StaticThereAndBackResource> {
      public:
        using inputBufferT     = buffer::HostToDevice<layout::Static<float>>;
        using temporaryBufferT = buffer::DeviceLocal<layout::Static<float>>;
        using outputBufferT    = buffer::DeviceToHost<layout::Static<float>>;

        static const uint32_t bufferSize = 8;

      public:
        StaticThereAndBackResource()                                            = default;
        StaticThereAndBackResource(const StaticThereAndBackResource& other)     = delete;
        StaticThereAndBackResource(StaticThereAndBackResource&& other) noexcept = default;

        ~StaticThereAndBackResource() override = default;

        StaticThereAndBackResource& operator=(const StaticThereAndBackResource& other) = delete;
        StaticThereAndBackResource&
        operator=(StaticThereAndBackResource&& other) noexcept = default;

        inputBufferT& getInput() {
            return this->input;
        }

        temporaryBufferT& getTemporary() {
            return this->temporary;
        }

        outputBufferT& getOutput() {
            return this->output;
        }

        template <typename CallableT>
        void forEachBuffer(CallableT callable) {
            callable(input);
            callable(temporary);
            callable(output);
        }

      private:
        inputBufferT     input{{{.itemCount = bufferSize, .set = 0, .binding = 0}}};
        temporaryBufferT temporary{{{.itemCount = bufferSize, .set = 0, .binding = 1}}};
        outputBufferT    output{{{.itemCount = bufferSize, .set = 0, .binding = 2}}};
    };

    class ComputeFrameworkTest : public ::testing::Test {
      public:
        static GLSL getShaderSource() {
            return GLSL{R"(

    #version 450

    #if (defined(SCALING_BUFFER_ARRAY) && (SCALING_BUFFER_ARRAY == 1))
        #extension GL_EXT_nonuniform_qualifier : require
    #endif

    #ifndef BATCH_SIZE
        #define BATCH_SIZE 16
    #endif

    layout(local_size_x = BATCH_SIZE , local_size_y = 1, local_size_z = 1) in;


    #ifndef BUFFER_0_SIZE
        #define BUFFER_0_SIZE 128
    #endif

    #ifndef BUFFER_1_SIZE
        #define BUFFER_1_SIZE 128
    #endif

    #ifndef BUFFER_2_SIZE
        #define BUFFER_2_SIZE 128
    #endif

    #if (defined(SCALING_BUFFER_ARRAY) && (SCALING_BUFFER_ARRAY == 1))
        #define BIND_BUFFER(setIndex, bindingIndex, typeName, varName)                  \
            layout(std430, set = setIndex, binding = bindingIndex) buffer typeName {    \
                float array[];                                                          \
            } varName [BATCH_SIZE]

        #define GET_ITEM_BUFFER_0(INDEX) \
            buffer0[gl_GlobalInvocationID.x].array[ INDEX ]

        #define GET_ITEM_BUFFER_1(INDEX) \
            buffer1[gl_GlobalInvocationID.x].array[ INDEX ]

        #define GET_ITEM_BUFFER_2(INDEX) \
            buffer2[gl_GlobalInvocationID.x].array[ INDEX ]

    #else
        #define BIND_BUFFER(setIndex, bindingIndex, typeName, varName)                  \
            layout(std430, set = setIndex, binding = bindingIndex) buffer typeName {    \
                float array[];                                                          \
            } varName

        #define GET_ITEM_BUFFER_0(INDEX) \
            buffer0.array[ gl_GlobalInvocationID.x * BATCH_SIZE + INDEX ]

        #define GET_ITEM_BUFFER_1(INDEX) \
            buffer1.array[ gl_GlobalInvocationID.x * BATCH_SIZE + INDEX ]

        #define GET_ITEM_BUFFER_2(INDEX) \
            buffer2.array[ gl_GlobalInvocationID.x * BATCH_SIZE + INDEX ]
    #endif

    BIND_BUFFER(0, 0, bufferT0, buffer0);
    BIND_BUFFER(0, 1, bufferT1, buffer1);
    BIND_BUFFER(0, 2, bufferT2, buffer2);

    void main() {
        for (int i = 0; i < BUFFER_0_SIZE; ++i) {
            GET_ITEM_BUFFER_1(i) = GET_ITEM_BUFFER_0(i);
        };
        for (int i = 0; i < BUFFER_1_SIZE; ++i) {
            GET_ITEM_BUFFER_2(i) = GET_ITEM_BUFFER_1(i);
        }
    }
        )"};
        }
    };

    TEST_F(ComputeFrameworkTest, StaticCopyThereAndBackOneInstanceTest) {

        const uint64_t                       batchSize = 1;
        std::shared_ptr<environment::Device> device    = environment::Device::create();
        std::shared_ptr<scaling::Base>       scalingPolicy =
            std::make_shared<scaling::LargeBuffer>(batchSize);

        shader::Static<StaticThereAndBackResource> shaderInstance{
            StaticThereAndBackResource{}, device, scalingPolicy};

        auto shaderGlsl = ComputeFrameworkTest::getShaderSource();
        shaderGlsl.updateMacroDefs(scalingPolicy->getImpliedMacroDefs());

        shaderInstance.prepare(shaderGlsl);
        shaderInstance.getResource().getInput().fillBuffers(
            [](uint32_t bufferIndex,
               float*   beginMappedData,
               float*   endMappedData,
               const layout::Static<float>& /*bufferLayout*/) {
                std::span<float> floatArray(beginMappedData, endMappedData);
                assert(!floatArray.empty());

                std::fill(floatArray.begin(), floatArray.end(), bufferIndex + 2);
            });
    }

    TEST_F(ComputeFrameworkTest, DynamicCopyThereAndBackOneInstanceTest) {
        const uint64_t batchSize  = 1;
        const uint64_t bufferSize = 128;

        std::shared_ptr<environment::Device> device = environment::Device::create();

        std::shared_ptr<scaling::Base> scalingPolicy =
            std::make_shared<scaling::LargeBuffer>(batchSize);

        shader::Dynamic shaderInstance{
            resources::Dynamic{
                std::vector<layout::Dynamic>{layout::Dynamic{
                    {.itemCount = bufferSize, .itemSize = sizeof(float), .set = 0, .binding = 0}}},
                std::vector<layout::Dynamic>{layout::Dynamic{
                    {.itemCount = bufferSize, .itemSize = sizeof(float), .set = 0, .binding = 1}}},
                std::vector<layout::Dynamic>{layout::Dynamic{
                    {.itemCount = bufferSize, .itemSize = sizeof(float), .set = 0, .binding = 2}}}},
            device,
            scalingPolicy};

        auto shaderGlsl = ComputeFrameworkTest::getShaderSource();
        shaderGlsl.updateMacroDefs(scalingPolicy->getImpliedMacroDefs());
        auto shaderSpirv = shaderGlsl.compile();

        shaderInstance.prepare(shaderSpirv);
        shaderInstance.getResource().getInput(0).fillBuffers(
            [](uint32_t bufferIndex,
               void*    beginMappedData,
               void*    endMappedData,
               const layout::Dynamic& /*bufferLayout*/) {
                auto* pBeginMappedData = static_cast<float*>(beginMappedData);
                auto* pEndMappedData   = static_cast<float*>(endMappedData);

                std::span<float> floatArray(pBeginMappedData, pEndMappedData);
                assert(!floatArray.empty());

                std::fill(floatArray.begin(), floatArray.end(), bufferIndex + 2);
            });
    }
} // namespace epseon::gpu::cpp
