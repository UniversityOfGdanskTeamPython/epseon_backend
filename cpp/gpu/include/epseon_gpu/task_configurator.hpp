
#pragma once
#include "epseon_gpu/compute_context.hpp"
#include "epseon_gpu/enums.hpp"
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            struct HardwareConfig {

              public:
                uint32_t potential_buffer_size = {};
                uint32_t group_size            = {};
                uint32_t allocation_block_size = {};

              public:
                HardwareConfig(
                    uint32_t potential_buffer_size_,
                    uint32_t group_size_,
                    uint32_t allocation_block_size_
                ) :
                    potential_buffer_size(potential_buffer_size_),
                    group_size(group_size_),
                    allocation_block_size(allocation_block_size_) {}

              public: /* Public destructor. */
                // Virtual destructor.
                virtual ~HardwareConfig() = default;

                virtual std::shared_ptr<HardwareConfig> shared_clone() const {
                    return std::make_shared<HardwareConfig>(*this);
                }

                virtual std::unique_ptr<HardwareConfig> unique_clone() const {
                    return std::make_unique<HardwareConfig>(*this);
                }
            };

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

            template <typename FP>
            class AlgorithmConfig {
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
                virtual std::shared_ptr<AlgorithmConfig<FP>> shared_clone() const = 0;
                virtual std::unique_ptr<AlgorithmConfig<FP>> unique_clone() const = 0;
            };

            template <typename FP>
            class VibwaAlgorithmConfig : public AlgorithmConfig<FP> {
              private:
                FP       integration_step          = {};
                FP       min_distance_to_asymptote = {};
                uint32_t min_level                 = {};
                uint32_t max_level                 = {};

              public: /* Public constructors. */
                VibwaAlgorithmConfig(
                    FP       integration_step_          = {},
                    FP       min_distance_to_asymptote_ = {},
                    uint32_t min_level_                 = {},
                    uint32_t max_level_                 = {}
                ) :
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
                virtual std::shared_ptr<AlgorithmConfig<FP>>
                shared_clone() const override {
                    return std::make_shared<VibwaAlgorithmConfig<FP>>(*this);
                }

                virtual std::unique_ptr<AlgorithmConfig<FP>>
                unique_clone() const override {
                    return std::make_unique<VibwaAlgorithmConfig<FP>>(*this);
                }
            };

            template <typename T>
            class RunningTask {};

            /* Builder for configuring GPU compute task. */
            template <typename FP>
            class TaskConfigurator {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              private:
                std::unique_ptr<HardwareConfig<FP>>  hardware_config  = {};
                std::unique_ptr<PotentialSource<FP>> potential_source = {};
                std::unique_ptr<AlgorithmConfig<FP>> algorithm_config = {};

              public: /* Public constructors. */
                // Default constructor
                TaskConfigurator()                                       = default;
                // Move constructor
                TaskConfigurator(TaskConfigurator&&) noexcept            = default;
                // Move assignment operator
                TaskConfigurator& operator=(TaskConfigurator&&) noexcept = default;

                // Copy constructor (deleted)
                TaskConfigurator(const TaskConfigurator& other) :
                    hardware_config(
                        other.hardware_config
                            ? std::move(other.hardware_config->unique_clone())
                            : nullptr
                    ),
                    potential_source(
                        other.potential_source
                            ? std::move(other.potential_source->unique_clone())
                            : nullptr
                    ),
                    algorithm_config(
                        other.algorithm_config
                            ? std::move(other.algorithm_config->unique_clone())
                            : nullptr
                    ) {}

                // Copy assignment operator
                TaskConfigurator& operator=(const TaskConfigurator& other) {
                    if (this != &other) {
                        hardware_config =
                            other.hardware_config
                                ? std::move(other.hardware_config->unique_clone())
                                : nullptr;
                        potential_source =
                            other.potential_source
                                ? std::move(other.potential_source->unique_clone())
                                : nullptr;
                        algorithm_config =
                            other.algorithm_config
                                ? std::move(other.algorithm_config->unique_clone())
                                : nullptr;
                    }
                    return *this;
                }

              public: /* Public destructor. */
                // Destructor
                ~TaskConfigurator() = default;

                /* Set hardware configuration for a GPU compute task. */
                TaskConfigurator&
                setHardwareConfig(std::unique_ptr<HardwareConfig<FP>> cfg) {
                    this->hardware_config.swap(cfg);
                    return *this;
                };

                /* Set potential data source configuration for GPU compute task. */
                TaskConfigurator&
                setPotentialSource(std::unique_ptr<PotentialSource<FP>> ps) {
                    this->potential_source.swap(ps);
                    return *this;
                };

                /* Set algorithm configuration for a GPU compute task. */
                TaskConfigurator&
                setAlgorithmConfig(std::unique_ptr<AlgorithmConfig<FP>> ac) {
                    this->algorithm_config.swap(ac);
                    return *this;
                };

                /* Check if this instance is fully configured, i.e. it has been assigned
                 * a valid hardware configuration, potential source and algorithm
                 * config.
                 */
                bool isConfigured() const {
                    return static_cast<bool>(this->hardware_config) &&
                           static_cast<bool>(this->potential_source) &&
                           static_cast<bool>(this->algorithm_config);
                };
            };

            template class TaskConfigurator<float>;
            template class TaskConfigurator<double>;

            typedef std::variant<TaskConfigurator<float>, TaskConfigurator<double>>
                TaskConfiguratorVariant;
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
