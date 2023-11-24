
#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/algorithms/algorithm.hpp"
#include "epseon/gpu/algorithms/vibwa.hpp"
#include <memory>
#include <type_traits>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class AlgorithmConfig
                : public std::enable_shared_from_this<AlgorithmConfig<FP>> {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              public: /* Public constructors. */
                // Default constructor.
                AlgorithmConfig() noexcept = default;

                // Copy constructor.
                AlgorithmConfig(const AlgorithmConfig&) noexcept = default;

                // Copy assignment operator.
                AlgorithmConfig& operator=(const AlgorithmConfig&) noexcept = default;

                // Move constructor.
                AlgorithmConfig(AlgorithmConfig&&) noexcept = default;

                // Move assignment operator.
                AlgorithmConfig& operator=(AlgorithmConfig&&) noexcept = default;

              public: /* Public destructor. */
                // Virtual destructor.
                virtual ~AlgorithmConfig() = default;

              public: /* Public methods. */
                virtual std::shared_ptr<Algorithm<FP>> getImplementation() const  = 0;
                virtual std::shared_ptr<AlgorithmConfig<FP>> shared_clone() const = 0;
                virtual std::unique_ptr<AlgorithmConfig<FP>> unique_clone() const = 0;
            };

            template <typename FP>
            class VibwaAlgorithm;

            template <typename FP>
            class VibwaAlgorithmConfig : public AlgorithmConfig<FP> {
              private:
                FP       mass_atom_0               = {};
                FP       mass_atom_1               = {};
                FP       integration_step          = {};
                FP       min_distance_to_asymptote = {};
                uint32_t min_level                 = {};
                uint32_t max_level                 = {};

              public: /* Public constructors. */
                VibwaAlgorithmConfig(
                    FP       mass_atom_0_               = {},
                    FP       mass_atom_1_               = {},
                    FP       integration_step_          = {},
                    FP       min_distance_to_asymptote_ = {},
                    uint32_t min_level_                 = {},
                    uint32_t max_level_                 = {}
                ) :
                    mass_atom_0(mass_atom_0_),
                    mass_atom_1(mass_atom_1_),
                    integration_step(integration_step_),
                    min_distance_to_asymptote(min_distance_to_asymptote_),
                    min_level(min_level_),
                    max_level(max_level_) {}

                // Default constructor.
                VibwaAlgorithmConfig() = default;

                // Copy constructor.
                VibwaAlgorithmConfig(const VibwaAlgorithmConfig&) noexcept = default;

                // Copy assignment operator.
                VibwaAlgorithmConfig&
                operator=(const VibwaAlgorithmConfig&) noexcept = default;

                // Move constructor.
                VibwaAlgorithmConfig(VibwaAlgorithmConfig&&) noexcept = default;

                // Move assignment operator.
                VibwaAlgorithmConfig&
                operator=(VibwaAlgorithmConfig&&) noexcept = default;

              public: /* Public destructor. */
                // Virtual destructor.
                virtual ~VibwaAlgorithmConfig() = default;

              public: /* Public methods. */
                virtual std::shared_ptr<Algorithm<FP>>
                getImplementation() const override {
                    return std::dynamic_pointer_cast<Algorithm<FP>>(
                        std::make_shared<VibwaAlgorithm<FP>>()
                    );
                }

                virtual std::shared_ptr<AlgorithmConfig<FP>>
                shared_clone() const override {
                    return std::make_shared<VibwaAlgorithmConfig<FP>>(*this);
                }

                virtual std::unique_ptr<AlgorithmConfig<FP>>
                unique_clone() const override {
                    return std::make_unique<VibwaAlgorithmConfig<FP>>(*this);
                }
            };
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
