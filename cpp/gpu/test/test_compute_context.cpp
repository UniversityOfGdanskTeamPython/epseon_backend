#include "epseon/gpu/compute_context.hpp"
#include "epseon/gpu/device_interface.hpp"
#include <gtest/gtest.h>
#include <regex>

namespace epseon::gpu::cpp {
    class ComputeContextTest : public ::testing::Test {};

    TEST_F(ComputeContextTest, TestConstruction) {
        auto ctx = ComputeContext::create();
    }

    TEST_F(ComputeContextTest, TestGetVulkanAPIVersion) {
        auto ctx = ComputeContext::create();
        if (ctx) {
            auto apiVersion = ctx->getVulkanAPIVersion();

            const std::regex versionRegex{R"([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)"};
            std::smatch      baseMatch;

            ASSERT_TRUE(std::regex_match(apiVersion, baseMatch, versionRegex));
        } else {
            ASSERT_TRUE(false && "Failed to create Vulkan interface.");
        }
    }

    TEST_F(ComputeContextTest, TestGetPhysicalDevicesInfo) {
        auto ctx         = ComputeContext::create();
        auto devicesInfo = ctx->getPhysicalDevicesInfo();
        ASSERT_NE(devicesInfo.size(), 0);
    }
} // namespace epseon::gpu::cpp
