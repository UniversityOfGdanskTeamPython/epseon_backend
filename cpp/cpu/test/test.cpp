#include "epseon_cpu/libcpu.hpp"
#include <gtest/gtest.h>

namespace LibCPU {

    class LibCPUTest : public ::testing::Test {};

    TEST_F(LibCPUTest, TestHello) {
        hello();
    }

} // namespace LibCPU