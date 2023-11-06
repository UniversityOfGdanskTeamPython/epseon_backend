#include "epseon_gpu/libgpu.hpp"
#include <gtest/gtest.h>

namespace LibGPU {

    class LibGPUTestA : public ::testing::Test {};

    TEST_F(LibGPUTestA, TestHello) {
        hello();
    }

} // namespace LibGPU
