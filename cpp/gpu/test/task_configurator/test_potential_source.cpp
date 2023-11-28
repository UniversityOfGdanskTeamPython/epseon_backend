#include "epseon/gpu/task_configurator/potential_source.hpp"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class PotentialFileLoaderTest : public ::testing::Test {
              protected:
                std::vector<std::string> fileNames      = {"file1.txt", "file2.txt"};
                PotentialFileLoader<FP>  loader_default = {};
                PotentialFileLoader<FP>  loader_custom  = {fileNames};
            };

            using MyTypes = ::testing::Types<float, double>;
            TYPED_TEST_SUITE(PotentialFileLoaderTest, MyTypes);

            TYPED_TEST(PotentialFileLoaderTest, DefaultConstructor) {
                EXPECT_TRUE(this->loader_default.get_potential_data().empty());
            }

            TYPED_TEST(PotentialFileLoaderTest, CustomConstructor) {
                auto data = this->loader_custom.get_potential_data();
                EXPECT_TRUE(data.empty()
                ); // Assuming get_potential_data() returns empty for this example
            }

            TYPED_TEST(PotentialFileLoaderTest, CopyConstructor) {
                PotentialFileLoader<TypeParam> loader_copy = this->loader_custom;
                auto                           data = loader_copy.get_potential_data();
                EXPECT_TRUE(data.empty());
            }

            TYPED_TEST(PotentialFileLoaderTest, MoveConstructor) {
                PotentialFileLoader<TypeParam> loader_moved(
                    std::move(this->loader_custom)
                );
                auto data = loader_moved.get_potential_data();
                EXPECT_TRUE(data.empty());
            }

            TYPED_TEST(PotentialFileLoaderTest, SharedCloneMethod) {
                auto cloned = this->loader_custom.shared_clone();
                EXPECT_NE(cloned, nullptr);
                auto potential_data = cloned->get_potential_data();
                EXPECT_TRUE(potential_data.empty());
            }

            TYPED_TEST(PotentialFileLoaderTest, UniqueCloneMethod) {
                auto cloned = this->loader_custom.unique_clone();
                EXPECT_NE(cloned, nullptr);
                auto potential_data = cloned->get_potential_data();
                EXPECT_TRUE(potential_data.empty());
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class MorsePotentialConfigTest : public ::testing::Test {
              protected:
                MorsePotentialConfig<FP> config_default = {};
                MorsePotentialConfig<FP> config_custom = {1.0, 2.0, 3.0, 0.1, 5.0, 100};
            };

            using MyTypes = ::testing::Types<float, double>;
            TYPED_TEST_SUITE(MorsePotentialConfigTest, MyTypes);

            TYPED_TEST(MorsePotentialConfigTest, DefaultConstructor) {
                EXPECT_EQ(this->config_default.getDissociationEnergy(), TypeParam{});
                EXPECT_EQ(
                    this->config_default.getEquilibriumBondDistance(), TypeParam{}
                );
                EXPECT_EQ(this->config_default.getWellWidth(), TypeParam{});
                EXPECT_EQ(this->config_default.getMinR(), TypeParam{});
                EXPECT_EQ(this->config_default.getMaxR(), TypeParam{});
                EXPECT_EQ(this->config_default.getPointCount(), 0u);
            }

            TYPED_TEST(MorsePotentialConfigTest, CustomConstructor) {
                EXPECT_EQ(this->config_custom.getDissociationEnergy(), TypeParam{1.0});
                EXPECT_EQ(
                    this->config_custom.getEquilibriumBondDistance(), TypeParam{2.0}
                );
                EXPECT_EQ(this->config_custom.getWellWidth(), TypeParam{3.0});
                EXPECT_EQ(this->config_custom.getMinR(), TypeParam{0.1});
                EXPECT_EQ(this->config_custom.getMaxR(), TypeParam{5.0});
                EXPECT_EQ(this->config_custom.getPointCount(), 100u);
            }

            TYPED_TEST(MorsePotentialConfigTest, CopyConstructor) {
                MorsePotentialConfig<TypeParam> copied_config(this->config_custom);
                EXPECT_EQ(
                    copied_config.getDissociationEnergy(),
                    this->config_custom.getDissociationEnergy()
                );
                EXPECT_EQ(
                    copied_config.getEquilibriumBondDistance(),
                    this->config_custom.getEquilibriumBondDistance()
                );
                EXPECT_EQ(
                    copied_config.getWellWidth(), this->config_custom.getWellWidth()
                );
                EXPECT_EQ(copied_config.getMinR(), this->config_custom.getMinR());
                EXPECT_EQ(copied_config.getMaxR(), this->config_custom.getMaxR());
                EXPECT_EQ(
                    copied_config.getPointCount(), this->config_custom.getPointCount()
                );
            }

            TYPED_TEST(MorsePotentialConfigTest, CopyAssignmentOperator) {
                MorsePotentialConfig<TypeParam> copied_config;
                copied_config = this->config_custom;
                EXPECT_EQ(
                    copied_config.getDissociationEnergy(),
                    this->config_custom.getDissociationEnergy()
                );
                EXPECT_EQ(
                    copied_config.getEquilibriumBondDistance(),
                    this->config_custom.getEquilibriumBondDistance()
                );
                EXPECT_EQ(
                    copied_config.getWellWidth(), this->config_custom.getWellWidth()
                );
                EXPECT_EQ(copied_config.getMinR(), this->config_custom.getMinR());
                EXPECT_EQ(copied_config.getMaxR(), this->config_custom.getMaxR());
                EXPECT_EQ(
                    copied_config.getPointCount(), this->config_custom.getPointCount()
                );
            }

            TYPED_TEST(MorsePotentialConfigTest, MoveConstructor) {
                // Create a copy of the custom configuration to compare after moving
                MorsePotentialConfig<TypeParam> config_before_move =
                    this->config_custom;

                // Move constructor
                MorsePotentialConfig<TypeParam> moved_config(
                    std::move(this->config_custom)
                );

                // Check if the moved-to object has the correct values
                EXPECT_EQ(
                    moved_config.getDissociationEnergy(),
                    config_before_move.getDissociationEnergy()
                );
                EXPECT_EQ(
                    moved_config.getEquilibriumBondDistance(),
                    config_before_move.getEquilibriumBondDistance()
                );
                EXPECT_EQ(
                    moved_config.getWellWidth(), config_before_move.getWellWidth()
                );
                EXPECT_EQ(moved_config.getMinR(), config_before_move.getMinR());
                EXPECT_EQ(moved_config.getMaxR(), config_before_move.getMaxR());
                EXPECT_EQ(
                    moved_config.getPointCount(), config_before_move.getPointCount()
                );
            }

            TYPED_TEST(MorsePotentialConfigTest, MoveAssignmentOperator) {
                // Create a copy of the custom configuration to compare after moving
                MorsePotentialConfig<TypeParam> config_before_move =
                    this->config_custom;

                // Move assignment operator
                MorsePotentialConfig<TypeParam> moved_config;
                moved_config = std::move(this->config_custom);

                // Check if the moved-to object has the correct values
                EXPECT_EQ(
                    moved_config.getDissociationEnergy(),
                    config_before_move.getDissociationEnergy()
                );
                EXPECT_EQ(
                    moved_config.getEquilibriumBondDistance(),
                    config_before_move.getEquilibriumBondDistance()
                );
                EXPECT_EQ(
                    moved_config.getWellWidth(), config_before_move.getWellWidth()
                );
                EXPECT_EQ(moved_config.getMinR(), config_before_move.getMinR());
                EXPECT_EQ(moved_config.getMaxR(), config_before_move.getMaxR());
                EXPECT_EQ(
                    moved_config.getPointCount(), config_before_move.getPointCount()
                );
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
