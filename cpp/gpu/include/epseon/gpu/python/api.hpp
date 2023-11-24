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
                    this->handle->start_worker();
                }

              public: /* Public methods. */
                std::string get_status_message() {
                    return "status";
                }

                /* Python API - Check if task is already finished. */
                bool is_done() {
                    return handle->is_done();
                }

                /* Python API - Cancel running task - will not result in immediate
                 * interrupt but rather will wait for response of worker thread.
                 *
                 * Returns true if the stop_source object has a stop-state and this
                 * invocation made a stop request, otherwise false
                 *
                 * Raises RuntimeError when internal worker doesn't exist.
                 */
                void cancel() {
                    handle->cancel();
                };

                /* Python API - Wait for worker thread to finish.
                 *
                 * Raises RuntimeError when internal worker doesn't exist.
                 */
                void wait() {
                    handle->wait();
                }
            };

            template class TaskHandle<float>;
            template class TaskHandle<double>;

            typedef TaskHandle<float>  TaskHandleFloat32;
            typedef TaskHandle<double> TaskHandleFloat64;

            typedef std::variant<TaskHandleFloat32, TaskHandleFloat64>
                TaskHandleVariant;

            class ComputeDeviceInterface {
              private:
                std::shared_ptr<cpp::ComputeDeviceInterface> device;

              public:
                ComputeDeviceInterface(std::shared_ptr<cpp::ComputeDeviceInterface>);
                /* Python API - Get builder instance for configuring GPU compute task.
                 */
                TaskConfigurator  get_task_configurator(std::string);
                /* Python API - Submit task for execution. Will raise RuntimeError upon
                 * receiving not fully configured TaskConfigurator. */
                TaskHandleVariant submit_task(const TaskConfigurator&);
            };

            class MorsePotentialConfig {
              private:
                cpp::MorsePotentialConfig<double> configuration;

              public: /* Public constructors. */
                // Member-wise constructor.
                MorsePotentialConfig(cpp::MorsePotentialConfig<double>&& configuration
                ) :
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
                MorsePotentialConfig&
                operator=(MorsePotentialConfig&&) noexcept = default;

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
            class TaskConfigurator {
              private:
                cpp::PrecisionType precision = cpp::PrecisionType::Float32;
                std::shared_ptr<cpp::TaskConfiguratorVariant> configurator = {};

              public:
                TaskConfigurator(cpp::PrecisionType, std::shared_ptr<cpp::TaskConfiguratorVariant>);
                // Copy constructor.
                TaskConfigurator(const TaskConfigurator&);
                // Copy assignment operator.
                TaskConfigurator& operator=(const TaskConfigurator&);
                // Move constructor.
                TaskConfigurator(TaskConfigurator&&) noexcept;
                // Move assignment operator.
                TaskConfigurator& operator=(TaskConfigurator&& other) noexcept;

              public:
                /* Python API - Set hardware configuration for a GPU compute task. */
                TaskConfigurator& set_hardware_config(uint32_t, uint32_t, uint32_t);

                /* Python API - Set potential data source configuration for GPU compute
                 * task. */
                TaskConfigurator&
                set_morse_potential(const std::vector<MorsePotentialConfig>&);

                /* Python API - Set algorithm configuration for a GPU compute task. */
                TaskConfigurator&
                set_vibwa_algorithm(double, double, double, double, uint32_t, uint32_t);

                /* Python API - Check if this instance is fully configured, i.e. it has
                 * been assigned a valid hardware configuration, potential source and
                 * algorithm config.
                 */
                bool is_configured() const;

                friend TaskHandleVariant
                ComputeDeviceInterface::submit_task(const TaskConfigurator&);
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
                /* Python API - Get interface for running algorithms on Vulkan devices.
                 */
                ComputeDeviceInterface               get_device_interface(uint32_t);
            };
        } // namespace python
    }     // namespace gpu
} // namespace epseon
