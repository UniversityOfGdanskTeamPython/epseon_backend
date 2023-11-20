
#include "epseon_gpu/task_configurator.hpp"
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace epseon {
    namespace gpu {
        namespace cpp {

            // Test Fixture for PotentialFileLoader
            template <typename FP>
            class PotentialFileLoaderTest : public ::testing::Test {
              protected:
                PotentialFileLoader<FP> loader;
            };

            // You might want to test with different floating-point types
            using MyTypes = ::testing::Types<float, double>;
            TYPED_TEST_SUITE(PotentialFileLoaderTest, MyTypes);

            // Test construction from vector of strings.
            TYPED_TEST(PotentialFileLoaderTest, VectorOfStringConstructor) {
                std::vector<std::string>       file_names = {"foo", "bar", "car"};
                PotentialFileLoader<TypeParam> loader{file_names};
            }

            // Test default constructor
            TYPED_TEST(PotentialFileLoaderTest, ArrayOfConstCharPtrConstructor) {
                std::array<const std::string, 3> file_names = {"foo", "bar", "car"};
                PotentialFileLoader<TypeParam>   loader{file_names};
            }

            // Test default constructor
            TYPED_TEST(PotentialFileLoaderTest, DefaultConstructor) {
                PotentialFileLoader<TypeParam> loader;
            }

            // Test copy constructor
            TYPED_TEST(PotentialFileLoaderTest, CopyConstructor) {
                PotentialFileLoader<TypeParam> loader1;
                PotentialFileLoader<TypeParam> loader2(loader1);
            }

            // Test copy assignment operator
            TYPED_TEST(PotentialFileLoaderTest, CopyAssignmentOperator) {
                PotentialFileLoader<TypeParam> loader1;
                PotentialFileLoader<TypeParam> loader2;
                loader2 = loader1;
            }

            // Test move constructor
            TYPED_TEST(PotentialFileLoaderTest, MoveConstructor) {
                PotentialFileLoader<TypeParam> loader1;
                PotentialFileLoader<TypeParam> loader2(std::move(loader1));
            }

            // Test move assignment operator
            TYPED_TEST(PotentialFileLoaderTest, MoveAssignmentOperator) {
                PotentialFileLoader<TypeParam> loader1;
                PotentialFileLoader<TypeParam> loader2;
                loader2 = std::move(loader1);
            }

            // Test get_potential_data method
            TYPED_TEST(PotentialFileLoaderTest, GetPotentialData) {
                auto data = this->loader.get_potential_data();
            }

            // Test shared_clone method
            TYPED_TEST(PotentialFileLoaderTest, SharedClone) {
                auto clone = this->loader.shared_clone();
            }

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
