#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/compute_context.hpp"
#include "epseon/gpu/device_interface.hpp"
#include "epseon/gpu/enums.hpp"
#include "epseon/gpu/task_configurator/algorithm_config.hpp"
#include "epseon/gpu/task_configurator/hardware_config.hpp"
#include "epseon/gpu/task_configurator/potential_source.hpp"
#include "epseon/gpu/task_configurator/task_configurator.hpp"
#include "epseon/gpu/task_handle.hpp"
#include "pybind11/pytypes.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace epseon {
    namespace gpu {
        namespace python {

            template <typename FP>
            class TaskHandle {
              private: /* Private members. */
                std::shared_ptr<cpp::TaskHandle<FP>> handle = {};

              public: /* Public constructors. */
                TaskHandle(std::shared_ptr<cpp::TaskHandle<FP>> handle_) :
                    handle(handle_) {
                    if (!this->handle->isRunning())
                        this->handle->startWorker();
                }

              public: /* Public methods. */
                std::string get_status_message() {
                    return "status";
                }

                /* Python API - Check if underlying worker thread finished its work.
                 * This doesn't check if thread even started, it will be false both if
                 * it is currently running and if it was never started. Use is_running()
                 * to clarify which of those is the case.
                 */
                bool is_done() {
                    return handle->isDone();
                }

                /* Python API - Check if task is still running. */
                bool is_running() {
                    return handle->isRunning();
                }

                /* Python API - Cancel running task - will not result in immediate
                 * interrupt but rather will wait for response of worker thread.
                 *
                 * Returns true if the stop_source object has a stop-state and this
                 * invocation made a stop request, otherwise false
                 */
                void cancel() {
                    handle->cancel();
                };

                /* Python API - Wait for worker thread to finish.
                 */
                void wait() {
                    handle->wait();
                }
            };

            template class TaskHandle<float>;
            template class TaskHandle<double>;

            typedef TaskHandle<float>  TaskHandleFloat32;
            typedef TaskHandle<double> TaskHandleFloat64;

            typedef std::variant<TaskHandleFloat32, TaskHandleFloat64> TaskHandleVariant;

            class MorsePotentialConfig {
              private:
                cpp::MorsePotentialConfig<double> configuration;

              public: /* Public constructors. */
                // Member-wise constructor.
                MorsePotentialConfig(cpp::MorsePotentialConfig<double>&& configuration) :
                    configuration(configuration) {}

                // Default constructor.
                MorsePotentialConfig() = default;

                // Copy constructor.
                MorsePotentialConfig(const MorsePotentialConfig&) = default;

                // Copy assignment operator.
                MorsePotentialConfig& operator=(const MorsePotentialConfig&) = default;

                // Move constructor.
                MorsePotentialConfig(MorsePotentialConfig&&) noexcept = default;

                // Move assignment operator.
                MorsePotentialConfig& operator=(MorsePotentialConfig&&) noexcept = default;

              public: /* Public methods. */
                static MorsePotentialConfig create(
                    double   dissociation_energy_,
                    double   equilibrium_bond_distance_,
                    double   well_width_,
                    double   min_r_,
                    double   max_r_,
                    uint32_t point_count_
                );

              public: /* Public methods. */
                const cpp::MorsePotentialConfig<double>& getConfiguration() const;
            };

            /* Python API - Wrapper class around TaskConfigurator class. */
            template <typename FP>
            class TaskConfigurator {
              private:
                std::shared_ptr<cpp::TaskConfigurator<FP>> configurator = {};

              public:
                TaskConfigurator(std::shared_ptr<cpp::TaskConfigurator<FP>> configurator_) :
                    configurator(configurator_) {}

                // Copy constructor.
                TaskConfigurator(const TaskConfigurator& other) noexcept :
                    configurator(other.configurator) {}

                // Copy assignment operator.
                TaskConfigurator& operator=(const TaskConfigurator& other) {
                    if (this != &other) {
                        configurator = other.configurator;
                    }
                    return *this;
                }

                // Move constructor.
                TaskConfigurator(TaskConfigurator&& other) noexcept :
                    configurator(std::move(other.configurator)) {}

                // Move assignment operator.
                TaskConfigurator& operator=(TaskConfigurator&& other) noexcept {
                    if (this != &other) {
                        configurator = std::move(other.configurator);
                    }
                    return *this;
                }

              public: /* Public methods. */
                /* Python API - Set hardware configuration for a GPU compute task. */
                TaskConfigurator& set_hardware_config(
                    uint32_t potential_buffer_size,
                    uint32_t group_size,
                    uint32_t allocation_block_size
                ) {
                    this->configurator->setHardwareConfig(std::make_shared<cpp::HardwareConfig<FP>>(
                        potential_buffer_size, group_size, allocation_block_size
                    ));
                    return *this;
                };

                /* Python API - Set potential data source configuration for GPU
                 * compute task. */
                TaskConfigurator&
                set_morse_potential(const std::vector<MorsePotentialConfig>& configurations) {
                    // Track used point count, all configs should have the same for now.
                    std::optional<uint32_t>                    point_count = std::nullopt;
                    // By giving from the start necessary vector size, we will avoid
                    // reallocating it multiple times.
                    std::vector<cpp::MorsePotentialConfig<FP>> configurations_cpp(
                        configurations.size()
                    );
                    // We will always get all floating point values in config as
                    // doubles, additionally we want to make sure users can't modify
                    // those values after assignment. Therefore we have to copy and
                    // possibly cast configuration values.
                    for (const auto& element : configurations) {
                        auto current_element_point_count =
                            element.getConfiguration().getPointCount();
                        // We need all buffer sizes to be the same to simplify GPU
                        // resource allocation. It is possible to implement this with no
                        // constraints on buffers sizes, but we are going for MVP now.
                        if (point_count.has_value() &&
                            point_count.value() != current_element_point_count) {
                            throw std::runtime_error(fmt::format(
                                "All Morse potentials must have same point "
                                "count, but previous ones had {} and current one "
                                "has.",
                                point_count.value(),
                                current_element_point_count
                            ));
                        } else {
                            point_count = {current_element_point_count};
                        }
                        // Insert new element into the back of the vector, explicitly
                        // casting to correct float type.
                        configurations_cpp.emplace_back(
                            static_cast<FP>(element.getConfiguration().getDissociationEnergy()),
                            static_cast<FP>(element.getConfiguration().getEquilibriumBondDistance()
                            ),
                            static_cast<FP>(element.getConfiguration().getWellWidth()),
                            static_cast<FP>(element.getConfiguration().getMinR()),
                            static_cast<FP>(element.getConfiguration().getMaxR()),
                            current_element_point_count
                        );
                    }
                    this->configurator->setPotentialSource(
                        std::make_shared<cpp::MorsePotentialGenerator<FP>>(
                            std::move(configurations_cpp)
                        )
                    );
                    return *this;
                }

                /* Python API - Set algorithm configuration for a GPU compute task.
                 */
                TaskConfigurator& set_vibwa_algorithm(
                    double   mass_atom_0,
                    double   mass_atom_1,
                    double   integration_step,
                    double   min_distance_to_asymptote,
                    uint32_t min_level,
                    uint32_t max_level
                ) {
                    this->configurator->setAlgorithmConfig(
                        std::make_shared<cpp::VibwaAlgorithmConfig<FP>>(
                            static_cast<FP>(mass_atom_0),
                            static_cast<FP>(mass_atom_1),
                            static_cast<FP>(integration_step),
                            static_cast<FP>(min_distance_to_asymptote),
                            min_level,
                            max_level
                        )
                    );
                    return *this;
                }

                /* Python API - Check if this instance is fully configured, i.e. it
                 * has been assigned a valid hardware configuration, potential
                 * source and algorithm config.
                 */
                bool is_configured() const {
                    return this->configurator->isConfigured();
                }

                std::shared_ptr<cpp::TaskConfigurator<FP>> getTaskConfigurator() const {
                    return this->configurator;
                }
            };

            template class TaskConfigurator<float>;
            template class TaskConfigurator<double>;

            typedef TaskConfigurator<float>  TaskConfiguratorFloat32;
            typedef TaskConfigurator<double> TaskConfiguratorFloat64;

            typedef std::variant<TaskConfiguratorFloat32, TaskConfiguratorFloat64>
                TaskConfiguratorVariant;

            class ComputeDeviceInterface {
              private:
                std::shared_ptr<cpp::ComputeDeviceInterface> device;

              public:
                ComputeDeviceInterface(std::shared_ptr<cpp::ComputeDeviceInterface>);
                /* Python API - Get builder instance for configuring GPU compute task.
                 */
                TaskConfiguratorVariant get_task_configurator(std::string);

                /* Python API - Submit task for execution. Will raise RuntimeError upon
                 * receiving not fully configured TaskConfigurator. */
                template <typename FP>
                TaskHandleVariant submit_task(const TaskConfigurator<FP>& task_config) {
                    if (!task_config.is_configured()) {
                        throw std::runtime_error("TaskConfigurator submitted for execution "
                                                 "before fully configured.");
                    }
                    auto config      = task_config.getTaskConfigurator();
                    auto task_handle = this->device->submitTask(config);

                    return TaskHandleVariant{// Namespaces specified explicitly to avoid confusion.
                                             epseon::gpu::python::TaskHandle<FP>{task_handle}
                    };
                }
            };

            class EpseonComputeContext {
              public:
                std::shared_ptr<cpp::ComputeContext> application = {};

              public:
                EpseonComputeContext(std::shared_ptr<cpp::ComputeContext>);

                static EpseonComputeContext create();

                /* Python API - returns Vulkan version extracted from VkInstance. */
                std::string                          get_vulkan_version();
                /* Python API - Get information about available physical devices. */
                std::vector<cpp::PhysicalDeviceInfo> get_physical_device_info();
                /* Python API - Get interface for running algorithms on Vulkan
                 * devices.
                 */
                ComputeDeviceInterface               get_device_interface(uint32_t);
            };
        } // namespace python
    }     // namespace gpu
} // namespace epseon
