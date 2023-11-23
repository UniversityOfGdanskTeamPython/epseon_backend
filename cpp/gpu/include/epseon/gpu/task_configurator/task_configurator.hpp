
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

                /* Get hardware configuration for a GPU compute task. */
                const HardwareConfig<FP>& getHardwareConfig() const {
                    return *this->hardware_config;
                };

                /* Set potential data source configuration for GPU compute task. */
                TaskConfigurator&
                setPotentialSource(std::unique_ptr<PotentialSource<FP>> ps) {
                    this->potential_source.swap(ps);
                    return *this;
                };

                /* Get potential data source configuration for GPU compute task. */
                const PotentialSource<FP>& getPotentialSource() const {
                    return *this->potential_source;
                };

                /* Set algorithm configuration for a GPU compute task. */
                TaskConfigurator&
                setAlgorithmConfig(std::unique_ptr<AlgorithmConfig<FP>> ac) {
                    this->algorithm_config.swap(ac);
                    return *this;
                };

                /* Get algorithm configuration for a GPU compute task. */
                const AlgorithmConfig<FP>& getAlgorithmConfig() const {
                    return *this->algorithm_config;
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
