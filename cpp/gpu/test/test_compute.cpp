#include "epseon/gpu/compute/environment.hpp"
#include "epseon/gpu/compute/predecl.hpp"
#include "epseon/gpu/compute/resources.hpp"
#include "epseon/gpu/compute/scaling.hpp"
#include "epseon/gpu/compute/shader.hpp"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <vector>

namespace epseon::gpu::cpp {

    class StaticThereAndBackResource : public resources::Static {
      public:
        using inputBufferT     = buffer::HostToDevice<layout::Static<float>>;
        using temporaryBufferT = buffer::DeviceLocal<layout::Static<float>>;
        using outputBufferT    = buffer::DeviceToHost<layout::Static<float>>;

        static const uint32_t bufferSize = 8;

        StaticThereAndBackResource() = default;

        inputBufferT& getInput() {
            return this->input;
        }

        temporaryBufferT& getTemporary() {
            return this->temporary;
        }

        outputBufferT& getOutput() {
            return this->output;
        }

      protected:
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

    class ComputeFrameworkTest : public ::testing::Test {};

    TEST_F(ComputeFrameworkTest, StaticCopyThereAndBackOneInstanceTest) {
        const uint64_t batchSize = 1;

        std::shared_ptr<environment::Device> device = environment::Device::create();
        std::shared_ptr<scaling::Base>       scalingPolicy =
            std::make_shared<scaling::LargeBuffer>(batchSize);

        shader::Static<StaticThereAndBackResource> shaderInstance{
            StaticThereAndBackResource{}, device, scalingPolicy};
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
} // namespace epseon::gpu::cpp
