
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
                virtual bool equals(const AlgorithmConfig<FP>& other) const       = 0;
                virtual std::shared_ptr<Algorithm<FP>> getImplementation() const  = 0;
                virtual std::shared_ptr<AlgorithmConfig<FP>> shared_clone() const = 0;
                virtual std::unique_ptr<AlgorithmConfig<FP>> unique_clone() const = 0;
            };

            template <typename FP>
            class VibwaAlgorithm;

            template <typename FP>
            class VibwaAlgorithmConfig : public AlgorithmConfig<FP> {
              private: /* Private members. */
                FP       mass_atom_0               = 0;
                FP       mass_atom_1               = 0;
                FP       integration_step          = 0;
                FP       min_distance_to_asymptote = 0;
                uint32_t min_level                 = 0;
                uint32_t max_level                 = 0;

              public: /* Public constructors. */
                VibwaAlgorithmConfig(
                    FP       mass_atom_0_,
                    FP       mass_atom_1_,
                    FP       integration_step_,
                    FP       min_distance_to_asymptote_,
                    uint32_t min_level_,
                    uint32_t max_level_
                ) :
                    mass_atom_0(mass_atom_0_),
                    mass_atom_1(mass_atom_1_),
                    integration_step(integration_step_),
                    min_distance_to_asymptote(min_distance_to_asymptote_),
                    min_level(min_level_),
                    max_level(max_level_) {}

                // Default constructor.
                VibwaAlgorithmConfig() noexcept = default;

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
                virtual bool equals(const AlgorithmConfig<FP>& other) const override {
                    const auto* otherCasted =
                        dynamic_cast<const VibwaAlgorithmConfig<FP>*>(&other);
                    if (otherCasted) {
                        return (
                            (this->mass_atom_0 == otherCasted->mass_atom_0) &&
                            (this->mass_atom_1 == otherCasted->mass_atom_1) &&
                            (this->integration_step == otherCasted->integration_step) &&
                            (this->min_distance_to_asymptote ==
                             otherCasted->min_distance_to_asymptote) &&
                            (this->min_level == otherCasted->min_level) &&
                            (this->max_level == otherCasted->max_level)
                        );
                    }
                    return false;
                }

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

              public: /* Public getters for members. */
                FP getMassAtom0() const {
                    return mass_atom_0;
                }

                FP getMassAtom1() const {
                    return mass_atom_1;
                }

                FP getIntegrationStep() const {
                    return integration_step;
                }

                FP getMinDistanceToAsymptote() const {
                    return min_distance_to_asymptote;
                }

                uint32_t getMinLevel() const {
                    return min_level;
                }

                uint32_t getMaxLevel() const {
                    return max_level;
                }
            };

            template <typename FP>
            bool
            operator==(const AlgorithmConfig<FP>& lhs, const AlgorithmConfig<FP>& rhs) {
                return lhs.equals(rhs);
            }

            template <typename FP>
            bool operator==(
                const VibwaAlgorithmConfig<FP>& lhs, const VibwaAlgorithmConfig<FP>& rhs
            ) {
                return lhs.equals(rhs);
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
