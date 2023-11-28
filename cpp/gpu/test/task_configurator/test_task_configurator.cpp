#include "epseon/gpu/task_configurator/algorithm_config.hpp"
#include "epseon/gpu/task_configurator/hardware_config.hpp"
#include "epseon/gpu/task_configurator/potential_source.hpp"
#include "gtest/gtest.h"
#include <memory>
#include <type_traits>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class TaskConfiguratorTest : public ::testing::Test {
              protected:
                TaskConfigurator<FP> configurator_default = {};
                TaskConfigurator<FP> configurator_custom{
                    std::make_shared<HardwareConfig<FP>>(),
                    std::make_shared<MorsePotentialGenerator<FP>>(),
                    std::make_shared<VibwaAlgorithmConfig<FP>>()
                };
            };

            using MyTypes = ::testing::Types<float, double>;
            TYPED_TEST_SUITE(TaskConfiguratorTest, MyTypes);

            TYPED_TEST(TaskConfiguratorTest, DefaultConstructor) {
                EXPECT_EQ(this->configurator_default.getHardwareConfig(), nullptr);
                EXPECT_EQ(this->configurator_default.getPotentialSource(), nullptr);
                EXPECT_EQ(this->configurator_default.getAlgorithmConfig(), nullptr);
                EXPECT_FALSE(this->configurator_default.isConfigured());
            }

            TYPED_TEST(TaskConfiguratorTest, CopyConstructor) {
                TaskConfigurator<TypeParam> copied_config(this->configurator_custom);
                ASSERT_EQ(
                    *copied_config.getHardwareConfig(),
                    *this->configurator_custom.getHardwareConfig()
                );
                ASSERT_EQ(
                    *copied_config.getPotentialSource(),
                    *this->configurator_custom.getPotentialSource()
                );
                ASSERT_EQ(
                    *copied_config.getAlgorithmConfig(),
                    *this->configurator_custom.getAlgorithmConfig()
                );
            }

            TYPED_TEST(TaskConfiguratorTest, CopyAssignmentOperator) {
                TaskConfigurator<TypeParam> copied_config;
                copied_config = this->configurator_custom;
                ASSERT_EQ(
                    *copied_config.getHardwareConfig(),
                    *this->configurator_custom.getHardwareConfig()
                );
                ASSERT_EQ(
                    *copied_config.getPotentialSource(),
                    *this->configurator_custom.getPotentialSource()
                );
                ASSERT_EQ(
                    *copied_config.getAlgorithmConfig(),
                    *this->configurator_custom.getAlgorithmConfig()
                );
            }

            TYPED_TEST(TaskConfiguratorTest, MoveConstructor) {
                TaskConfigurator<TypeParam> moved_config(
                    std::move(this->configurator_custom)
                );
                EXPECT_TRUE(moved_config.isConfigured()
                ); // This check depends on the state of configurator_custom
            }

            TYPED_TEST(TaskConfiguratorTest, MoveAssignmentOperator) {
                TaskConfigurator<TypeParam> moved_config;
                moved_config = std::move(this->configurator_custom);
                EXPECT_TRUE(moved_config.isConfigured()
                ); // This check depends on the state of configurator_custom
            }

            TYPED_TEST(TaskConfiguratorTest, SetHardwareConfig) {
                auto hw_config = std::make_shared<HardwareConfig<TypeParam>>();
                this->configurator_default.setHardwareConfig(hw_config);
                ASSERT_EQ(*this->configurator_default.getHardwareConfig(), *hw_config);
            }

            TYPED_TEST(TaskConfiguratorTest, SetPotentialSource) {
                auto ps = std::make_shared<MorsePotentialGenerator<TypeParam>>();
                this->configurator_default.setPotentialSource(ps);
                auto ps2 =
                    std::dynamic_pointer_cast<MorsePotentialGenerator<TypeParam>>(
                        this->configurator_default.getPotentialSource()
                    );
                ASSERT_EQ(*ps2, *ps);
            }

            TYPED_TEST(TaskConfiguratorTest, SetAlgorithmConfig) {
                auto ac = std::make_shared<VibwaAlgorithmConfig<TypeParam>>();
                this->configurator_default.setAlgorithmConfig(ac);
                auto algorithm =
                    std::dynamic_pointer_cast<VibwaAlgorithmConfig<TypeParam>>(
                        this->configurator_default.getAlgorithmConfig()
                    );
                ASSERT_EQ(*algorithm, *ac);
            }

            TYPED_TEST(TaskConfiguratorTest, IsConfigured) {
                auto hw_config = std::make_shared<HardwareConfig<TypeParam>>();
                auto ps        = std::make_shared<MorsePotentialGenerator<TypeParam>>();
                auto ac        = std::make_shared<VibwaAlgorithmConfig<TypeParam>>();
                this->configurator_default.setHardwareConfig(hw_config)
                    .setPotentialSource(ps)
                    .setAlgorithmConfig(ac);
                EXPECT_TRUE(this->configurator_default.isConfigured());
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
