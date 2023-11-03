#include "epseon_gpu/libgpu.hpp"
#include <gtest/gtest.h>

namespace LibGPU {

    class LibGPUTest : public ::testing::Test {};

    TEST_F(LibGPUTest, TestHello) {
        hello();
    }

} // namespace LibGPU