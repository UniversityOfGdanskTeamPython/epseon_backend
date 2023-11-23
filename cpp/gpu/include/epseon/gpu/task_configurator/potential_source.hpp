
#pragma once

#include "epseon/gpu/predecl.hpp"
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class PotentialSource {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              public: /* Public constructors. */
                // Default constructor.
                PotentialSource() noexcept = default;

                // Copy constructor.
                PotentialSource(const PotentialSource&) noexcept = default;

                // Copy assignment operator.
                PotentialSource& operator=(const PotentialSource&) noexcept = default;

                // Move constructor.
                PotentialSource(PotentialSource&&) noexcept = default;

                // Move assignment operator.
                PotentialSource& operator=(PotentialSource&&) noexcept = default;

              public: /* Public destructor. */
                // Virtual destructor.
                virtual ~PotentialSource() = default;

              public: /* Public methods. */
                virtual std::vector<std::vector<FP>>         get_potential_data() = 0;
                virtual std::shared_ptr<PotentialSource<FP>> shared_clone() const = 0;
                virtual std::unique_ptr<PotentialSource<FP>> unique_clone() const = 0;
            };

            template <typename FP>
            class PotentialFileLoader : public PotentialSource<FP> {
              private:
                std::vector<std::string> file_names = {};

              public: /* Public constructors. */
                // File paths constructor.
                PotentialFileLoader(const std::span<const std::string> file_names) :
                    file_names([file_names]() {
                        std::vector<std::string> local_file_names(file_names.size());
                        for (uint64_t i = 0; i < file_names.size(); i++) {
                            local_file_names[i] = file_names[i];
                        }
                        return std::move(local_file_names);
                    }()) {}

                // Default constructor.
                PotentialFileLoader() noexcept = default;

                // Copy constructor.
                PotentialFileLoader(const PotentialFileLoader&) noexcept = default;

                // Copy assignment operator.
                PotentialFileLoader&
                operator=(const PotentialFileLoader&) noexcept = default;

                // Move constructor.
                PotentialFileLoader(PotentialFileLoader&&) noexcept = default;

                // Move assignment operator.
                PotentialFileLoader&
                operator=(PotentialFileLoader&&) noexcept = default;

              public: /* Public destructor. */
                // Virtual destructor.
                virtual ~PotentialFileLoader() = default;

              public: /* Public methods. */
                virtual std::vector<std::vector<FP>> get_potential_data() override {
                    return {};
                }

                virtual std::shared_ptr<PotentialSource<FP>>
                shared_clone() const override {
                    return std::make_shared<PotentialFileLoader<FP>>(*this);
                }

                virtual std::unique_ptr<PotentialSource<FP>>
                unique_clone() const override {
                    return std::make_unique<PotentialFileLoader<FP>>(*this);
                }
            };

            template <typename FP>
            class MorsePotentialConfig {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              private:
                FP       dissociation_energy       = {};
                FP       equilibrium_bond_distance = {};
                FP       well_width                = {};
                FP       min_r                     = {};
                FP       max_r                     = {};
                uint32_t point_count               = {};

              public: /* Public constructors. */
                // Member-wise constructor.
                MorsePotentialConfig(
                    FP       dissociation_energy_,
                    FP       equilibrium_bond_distance_,
                    FP       well_width_,
                    FP       min_r_,
                    FP       max_r_,
                    uint32_t point_count_
                ) :
                    dissociation_energy(dissociation_energy_),
                    equilibrium_bond_distance(equilibrium_bond_distance_),
                    well_width(well_width_),
                    min_r(min_r_),
                    max_r(max_r_),
                    point_count(point_count_) {}

                // Default constructor.
                MorsePotentialConfig() = default;

                // Copy constructor.
                MorsePotentialConfig(const MorsePotentialConfig&) = default;

                // Copy assignment operator.
                MorsePotentialConfig& operator=(const MorsePotentialConfig&) = default;

                // Move constructor.
                MorsePotentialConfig(MorsePotentialConfig&&) noexcept = default;

                // Move assignment operator.
                MorsePotentialConfig&
                operator=(MorsePotentialConfig&&) noexcept = default;

              public: /* Public methods. */
                template <typename FP_>
                FP_ getDissociationEnergy() const {
                    return static_cast<FP_>(this->dissociation_energy);
                }

                template <typename FP_>
                FP_ getEquilibriumBondDistance() const {
                    return static_cast<FP_>(this->equilibrium_bond_distance);
                }

                template <typename FP_>
                FP_ getWellWidth() const {
                    return static_cast<FP_>(this->well_width);
                }

                template <typename FP_>
                FP_ getMinR() const {
                    return static_cast<FP_>(this->min_r);
                }

                template <typename FP_>
                FP_ getMaxR() const {
                    return static_cast<FP_>(this->max_r);
                }

                uint32_t getPointCount() const {
                    return this->point_count;
                }
            };

            template <typename FP>
            class MorsePotentialGenerator : public PotentialSource<FP> {
              public:
                std::vector<MorsePotentialConfig<FP>> configurations = {};

              public: /* Public constructors. */
                // Member-wise constructor.
                MorsePotentialGenerator(
                    std::vector<MorsePotentialConfig<FP>>&& configurations_
                ) :
                    configurations(configurations_) {}

                // Default constructor.
                MorsePotentialGenerator() = default;

                // Copy constructor.
                MorsePotentialGenerator(const MorsePotentialGenerator&) = default;

                // Copy assignment operator.
                MorsePotentialGenerator&
                operator=(const MorsePotentialGenerator&) = default;

                // Move constructor.
                MorsePotentialGenerator(MorsePotentialGenerator&&) noexcept = default;

                // Move assignment operator.
                MorsePotentialGenerator&
                operator=(MorsePotentialGenerator&&) noexcept = default;

              public: /* Public destructor. */
                // Virtual destructor.
                virtual ~MorsePotentialGenerator() = default;

              public: /* Public methods. */
                virtual std::vector<std::vector<FP>> get_potential_data() override {
                    return {};
                }

                virtual std::shared_ptr<PotentialSource<FP>>
                shared_clone() const override {
                    return std::make_shared<MorsePotentialGenerator<FP>>(*this);
                }

                virtual std::unique_ptr<PotentialSource<FP>>
                unique_clone() const override {
                    return std::make_unique<MorsePotentialGenerator<FP>>(*this);
                }
            };
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
