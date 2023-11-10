#include "epseon_gpu/libgpu.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <regex>

namespace epseon {
    namespace gpu {
        namespace cpp {
            class LibGPUTest : public ::testing::Test {};

            TEST_F(LibGPUTest, TestVulkanApplicationConstruction) {
                auto va = VulkanApplication::create();
                if (va) {
                    auto apiVersion = va->getVulkanAPIVersion();

                    const std::regex versionRegex{"[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+"};
                    std::smatch      baseMatch;

                    ASSERT_TRUE(std::regex_match(apiVersion, baseMatch, versionRegex));
                } else {
                    ASSERT_TRUE(false && "Failed to create Vulkan interface.");
                }
            }

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
