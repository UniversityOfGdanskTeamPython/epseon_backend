#include "epseon/gpu/compute/environment.hpp"
#include "epseon/gpu/compute/predecl.hpp"
#include "epseon/gpu/compute/resources.hpp"
#include "epseon/gpu/compute/scaling.hpp"
#include "epseon/gpu/compute/shader.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace epseon::gpu::cpp {

    class StaticThereAndBackResource : public resources::Static {
      public:
        static const uint32_t bufferSize = 8;

        StaticThereAndBackResource() = default;

      protected:
        template <typename CallableT>
        void forEachBuffer(CallableT callable) {
            callable(input);
            callable(temporary);
            callable(output);
        }

      private:
        buffer::HostToDevice<layout::Static<float>> input{
            {{.itemCount = bufferSize, .set = 0, .binding = 0}}};
        buffer::DeviceLocal<layout::Static<float>> temporary{
            {{.itemCount = bufferSize, .set = 0, .binding = 1}}};
        buffer::DeviceToHost<layout::Static<float>> output{
            {{.itemCount = bufferSize, .set = 0, .binding = 2}}};
    };

    class ComputeFrameworkTest : public ::testing::Test {};

    TEST_F(ComputeFrameworkTest, StaticCopyThereAndBackOneInstanceTest) {
        const uint64_t batchSize  = 1;

        std::shared_ptr<environment::Device> device = environment::Device::create();
        std::shared_ptr<scaling::Base>       scalingPolicy =
            std::make_shared<scaling::LargeBuffer>(batchSize);

        shader::Static<VibwaResources> shaderInstance{VibwaResources{}, device, scalingPolicy};
        shaderInstance.getResource().;
    }
} // namespace epseon::gpu::cpp
