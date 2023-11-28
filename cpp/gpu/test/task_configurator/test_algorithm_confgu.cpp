#include "epseon/gpu/task_configurator/algorithm_config.hpp" // Include the appropriate header
#include "gtest/gtest.h"
#include <memory>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class VibwaAlgorithmConfigTest : public ::testing::Test {
              protected:
                VibwaAlgorithmConfig<FP> config_default = {};
                VibwaAlgorithmConfig<FP> config_custom  = {1.0, 2.0, 0.1, 0.05, 10, 20};
            };

            using MyTypes = ::testing::Types<float, double>;
            TYPED_TEST_SUITE(VibwaAlgorithmConfigTest, MyTypes);

            TYPED_TEST(VibwaAlgorithmConfigTest, DefaultConstructor) {
                EXPECT_EQ(this->config_default.getMassAtom0(), TypeParam{});
                EXPECT_EQ(this->config_default.getMassAtom1(), TypeParam{});
                EXPECT_EQ(this->config_default.getIntegrationStep(), TypeParam{});
                EXPECT_EQ(
                    this->config_default.getMinDistanceToAsymptote(), TypeParam{}
                );
                EXPECT_EQ(this->config_default.getMinLevel(), 0u);
                EXPECT_EQ(this->config_default.getMaxLevel(), 0u);
            }

            TYPED_TEST(VibwaAlgorithmConfigTest, ParameterizedConstructor) {
                EXPECT_EQ(this->config_custom.getMassAtom0(), TypeParam{1.0});
                EXPECT_EQ(this->config_custom.getMassAtom1(), TypeParam{2.0});
                EXPECT_EQ(this->config_custom.getIntegrationStep(), TypeParam{0.1});
                EXPECT_EQ(
                    this->config_custom.getMinDistanceToAsymptote(), TypeParam{0.05}
                );
                EXPECT_EQ(this->config_custom.getMinLevel(), 10u);
                EXPECT_EQ(this->config_custom.getMaxLevel(), 20u);
            }

            // Tests for the copy constructor
            TYPED_TEST(VibwaAlgorithmConfigTest, CopyConstructor) {
                VibwaAlgorithmConfig<TypeParam> copied_config(this->config_custom);
                EXPECT_EQ(
                    copied_config.getMassAtom0(), this->config_custom.getMassAtom0()
                );
                EXPECT_EQ(
                    copied_config.getMassAtom1(), this->config_custom.getMassAtom1()
                );
                EXPECT_EQ(
                    copied_config.getIntegrationStep(),
                    this->config_custom.getIntegrationStep()
                );
                EXPECT_EQ(
                    copied_config.getMinDistanceToAsymptote(),
                    this->config_custom.getMinDistanceToAsymptote()
                );
                EXPECT_EQ(
                    copied_config.getMinLevel(), this->config_custom.getMinLevel()
                );
                EXPECT_EQ(
                    copied_config.getMaxLevel(), this->config_custom.getMaxLevel()
                );
            }

            // Tests for the copy assignment operator
            TYPED_TEST(VibwaAlgorithmConfigTest, CopyAssignmentOperator) {
                VibwaAlgorithmConfig<TypeParam> copied_config;
                copied_config = this->config_custom;
                EXPECT_EQ(
                    copied_config.getMassAtom0(), this->config_custom.getMassAtom0()
                );
                EXPECT_EQ(
                    copied_config.getMassAtom1(), this->config_custom.getMassAtom1()
                );
                EXPECT_EQ(
                    copied_config.getIntegrationStep(),
                    this->config_custom.getIntegrationStep()
                );
                EXPECT_EQ(
                    copied_config.getMinDistanceToAsymptote(),
                    this->config_custom.getMinDistanceToAsymptote()
                );
                EXPECT_EQ(
                    copied_config.getMinLevel(), this->config_custom.getMinLevel()
                );
                EXPECT_EQ(
                    copied_config.getMaxLevel(), this->config_custom.getMaxLevel()
                );
            }

            // Tests for the move constructor
            TYPED_TEST(VibwaAlgorithmConfigTest, MoveConstructor) {
                VibwaAlgorithmConfig<TypeParam> moved_config(
                    std::move(this->config_custom)
                );
                EXPECT_EQ(moved_config.getMassAtom0(), TypeParam{1.0});
                EXPECT_EQ(moved_config.getMassAtom1(), TypeParam{2.0});
                EXPECT_EQ(moved_config.getIntegrationStep(), TypeParam{0.1});
                EXPECT_EQ(moved_config.getMinDistanceToAsymptote(), TypeParam{0.05});
                EXPECT_EQ(moved_config.getMinLevel(), 10u);
                EXPECT_EQ(moved_config.getMaxLevel(), 20u);
                // Optionally, verify that the moved-from object is in a valid but
                // unspecified state
            }

            // Tests for the move assignment operator
            TYPED_TEST(VibwaAlgorithmConfigTest, MoveAssignmentOperator) {
                VibwaAlgorithmConfig<TypeParam> moved_config;
                moved_config = std::move(this->config_custom);
                EXPECT_EQ(moved_config.getMassAtom0(), TypeParam{1.0});
                EXPECT_EQ(moved_config.getMassAtom1(), TypeParam{2.0});
                EXPECT_EQ(moved_config.getIntegrationStep(), TypeParam{0.1});
                EXPECT_EQ(moved_config.getMinDistanceToAsymptote(), TypeParam{0.05});
                EXPECT_EQ(moved_config.getMinLevel(), 10u);
                EXPECT_EQ(moved_config.getMaxLevel(), 20u);
                // Optionally, verify that the moved-from object is in a valid but
                // unspecified state
            }

            TYPED_TEST(VibwaAlgorithmConfigTest, SharedCloneMethod) {
                auto cloned = this->config_custom.shared_clone();
                EXPECT_NE(cloned, nullptr);
                auto cloned_cast =
                    std::dynamic_pointer_cast<VibwaAlgorithmConfig<TypeParam>>(cloned);
                EXPECT_NE(cloned_cast, nullptr);
                EXPECT_EQ(cloned_cast->getMassAtom0(), TypeParam{1.0});
                EXPECT_EQ(cloned_cast->getMassAtom1(), TypeParam{2.0});
                EXPECT_EQ(cloned_cast->getIntegrationStep(), TypeParam{0.1});
                EXPECT_EQ(cloned_cast->getMinDistanceToAsymptote(), TypeParam{0.05});
                EXPECT_EQ(cloned_cast->getMinLevel(), 10u);
                EXPECT_EQ(cloned_cast->getMaxLevel(), 20u);
            }

            TYPED_TEST(VibwaAlgorithmConfigTest, UniqueCloneMethod) {
                auto cloned = this->config_custom.unique_clone();
                EXPECT_NE(cloned, nullptr);
                auto cloned_cast =
                    static_cast<VibwaAlgorithmConfig<TypeParam>*>(cloned.get());
                EXPECT_NE(cloned_cast, nullptr);
                EXPECT_EQ(cloned_cast->getMassAtom0(), TypeParam{1.0});
                EXPECT_EQ(cloned_cast->getMassAtom1(), TypeParam{2.0});
                EXPECT_EQ(cloned_cast->getIntegrationStep(), TypeParam{0.1});
                EXPECT_EQ(cloned_cast->getMinDistanceToAsymptote(), TypeParam{0.05});
                EXPECT_EQ(cloned_cast->getMinLevel(), 10u);
                EXPECT_EQ(cloned_cast->getMaxLevel(), 20u);
            }

            TYPED_TEST(VibwaAlgorithmConfigTest, GetImplementationMethod) {
                auto implementation = this->config_default.getImplementation();
                EXPECT_NE(implementation, nullptr);
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
