
#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/task_configurator/algorithm_config.hpp"
#include "epseon/gpu/task_configurator/hardware_config.hpp"
#include "epseon/gpu/task_configurator/potential_source.hpp"
#include <memory>
#include <type_traits>
#include <variant>

namespace epseon {
    namespace gpu {
        namespace cpp {

            /* Builder for configuring GPU compute task. */
            template <typename FP>
            class TaskConfigurator
                : public std::enable_shared_from_this<TaskConfigurator<FP>> {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              private:
                std::shared_ptr<HardwareConfig<FP>>  hardware_config  = {};
                std::shared_ptr<PotentialSource<FP>> potential_source = {};
                std::shared_ptr<AlgorithmConfig<FP>> algorithm_config = {};

              public: /* Public constructors. */
                // Parametrized constructor
                TaskConfigurator(
                    std::shared_ptr<HardwareConfig<FP>>  hardware_config_,
                    std::shared_ptr<PotentialSource<FP>> potential_source_,
                    std::shared_ptr<AlgorithmConfig<FP>> algorithm_config_

                ) :
                    hardware_config(hardware_config_),
                    potential_source(potential_source_),
                    algorithm_config(algorithm_config_){};

                // Default constructor
                TaskConfigurator() = default;
                // Move constructor
                TaskConfigurator(TaskConfigurator&& other) noexcept :
                    hardware_config(std::move(other.hardware_config)),
                    potential_source(std::move(other.potential_source)),
                    algorithm_config(std::move(other.algorithm_config)){};

                // Move assignment operator
                TaskConfigurator& operator=(TaskConfigurator&& other) noexcept {
                    if (this != &other) {
                        hardware_config  = std::move(other.hardware_config);
                        potential_source = std::move(other.potential_source);
                        algorithm_config = std::move(other.algorithm_config);
                    }
                    return *this;
                };

                // Copy constructor (deleted)
                TaskConfigurator(const TaskConfigurator& other) :
                    hardware_config(
                        other.hardware_config
                            ? std::move(other.hardware_config->shared_clone())
                            : nullptr
                    ),
                    potential_source(
                        other.potential_source
                            ? std::move(other.potential_source->shared_clone())
                            : nullptr
                    ),
                    algorithm_config(
                        other.algorithm_config
                            ? std::move(other.algorithm_config->shared_clone())
                            : nullptr
                    ) {}

                // Copy assignment operator
                TaskConfigurator& operator=(const TaskConfigurator& other) {
                    if (this != &other) {
                        hardware_config =
                            other.hardware_config
                                ? std::move(other.hardware_config->shared_clone())
                                : nullptr;
                        potential_source =
                            other.potential_source
                                ? std::move(other.potential_source->shared_clone())
                                : nullptr;
                        algorithm_config =
                            other.algorithm_config
                                ? std::move(other.algorithm_config->shared_clone())
                                : nullptr;
                    }
                    return *this;
                }

              public: /* Public destructor. */
                ~TaskConfigurator() {}

              public: /* Public methods. */
                /* Set hardware configuration for a GPU compute task. */
                TaskConfigurator&
                setHardwareConfig(std::shared_ptr<HardwareConfig<FP>> cfg) {
                    this->hardware_config = cfg->shared_clone();
                    return *this;
                };

                /* Get hardware configuration for a GPU compute task. */
                const std::shared_ptr<HardwareConfig<FP>> getHardwareConfig() const {
                    return this->hardware_config;
                };

                /* Set potential data source configuration for GPU compute task. */
                TaskConfigurator&
                setPotentialSource(std::shared_ptr<PotentialSource<FP>> ps) {
                    this->potential_source = ps->shared_clone();
                    return *this;
                };

                /* Get potential data source configuration for GPU compute task. */
                const std::shared_ptr<PotentialSource<FP>> getPotentialSource() const {
                    return this->potential_source;
                };

                /* Set algorithm configuration for a GPU compute task. */
                TaskConfigurator&
                setAlgorithmConfig(std::shared_ptr<AlgorithmConfig<FP>> ac) {
                    this->algorithm_config = ac->shared_clone();
                    return *this;
                };

                /* Get algorithm configuration for a GPU compute task. */
                const std::shared_ptr<AlgorithmConfig<FP>> getAlgorithmConfig() const {
                    return this->algorithm_config;
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
