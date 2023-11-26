#include "epseon/gpu/task_configurator/hardware_config.hpp"
#include <gtest/gtest.h>

namespace epseon {
    namespace gpu {
        namespace cpp {
            using HardwareConfigTypes = ::testing::Types<
                epseon::gpu::cpp::HardwareConfig<float>,
                epseon::gpu::cpp::HardwareConfig<double>>;

            template <typename T>
            class HardwareConfigTest : public ::testing::Test {
              protected:
                std::shared_ptr<T> config;

                void SetUp() override {
                    // Initialize with hypothetical values
                    config = std::make_shared<T>(100, 200, 300);
                }
            };

            TYPED_TEST_SUITE(HardwareConfigTest, HardwareConfigTypes);

            TYPED_TEST(HardwareConfigTest, ConstructorInitializesValues) {
                EXPECT_EQ(this->config->potential_buffer_size, 100);
                EXPECT_EQ(this->config->group_size, 200);
                EXPECT_EQ(this->config->allocation_block_size, 300);
            }

            TYPED_TEST(HardwareConfigTest, SharedCloneCreatesCorrectCopy) {
                auto clone = this->config->shared_clone();
                EXPECT_EQ(
                    clone->potential_buffer_size, this->config->potential_buffer_size
                );
                EXPECT_EQ(clone->group_size, this->config->group_size);
                EXPECT_EQ(
                    clone->allocation_block_size, this->config->allocation_block_size
                );
            }

            TYPED_TEST(HardwareConfigTest, UniqueCloneCreatesCorrectCopy) {
                auto clone = this->config->unique_clone();
                EXPECT_EQ(
                    clone->potential_buffer_size, this->config->potential_buffer_size
                );
                EXPECT_EQ(clone->group_size, this->config->group_size);
                EXPECT_EQ(
                    clone->allocation_block_size, this->config->allocation_block_size
                );
            }

            TYPED_TEST(HardwareConfigTest, DefaultCopyConstructor) {
                auto copiedConfig = *this->config;
                EXPECT_EQ(
                    copiedConfig.potential_buffer_size,
                    this->config->potential_buffer_size
                );
                EXPECT_EQ(copiedConfig.group_size, this->config->group_size);
                EXPECT_EQ(
                    copiedConfig.allocation_block_size,
                    this->config->allocation_block_size
                );
            }

            TYPED_TEST(HardwareConfigTest, DefaultMoveConstructor) {
                TypeParam tempConfig(
                    100, 200, 300
                ); // Create a temporary config for moving
                TypeParam movedConfig(std::move(tempConfig));
                EXPECT_EQ(movedConfig.potential_buffer_size, 100);
                EXPECT_EQ(movedConfig.group_size, 200);
                EXPECT_EQ(movedConfig.allocation_block_size, 300);
            }

            TYPED_TEST(HardwareConfigTest, CopyAssignmentOperator) {
                TypeParam otherConfig(400, 500, 600); // Different initialization
                otherConfig = *this->config;
                EXPECT_EQ(
                    otherConfig.potential_buffer_size,
                    this->config->potential_buffer_size
                );
                EXPECT_EQ(otherConfig.group_size, this->config->group_size);
                EXPECT_EQ(
                    otherConfig.allocation_block_size,
                    this->config->allocation_block_size
                );
            }

            TYPED_TEST(HardwareConfigTest, MoveAssignmentOperator) {
                TypeParam tempConfig(
                    100, 200, 300
                ); // Create a temporary config for moving
                TypeParam otherConfig(400, 500, 600); // Different initialization
                otherConfig = std::move(tempConfig);
                EXPECT_EQ(otherConfig.potential_buffer_size, 100);
                EXPECT_EQ(otherConfig.group_size, 200);
                EXPECT_EQ(otherConfig.allocation_block_size, 300);
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
