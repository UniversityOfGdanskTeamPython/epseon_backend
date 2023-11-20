#pragma once

#include "epseon/libepseon.hpp"
#include "epseon_gpu/compute_context.hpp"
#include "epseon_gpu/device_interface.hpp"
#include "epseon_gpu/enums.hpp"
#include "epseon_gpu/task_configurator.hpp"
#include "epseon_gpu/task_handle.hpp"
#include "pybind11/pytypes.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace epseon {
    namespace gpu {
        namespace python {

            class TaskConfigurator;

            template <typename FP>
            class TaskHandle {
              private:
                std::shared_ptr<cpp::TaskHandle<FP>> handle = {};

              public:
                TaskHandle(std::shared_ptr<cpp::TaskHandle<FP>> handle_) :
                    handle(handle_) {}
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
                TaskConfigurator&
                    set_hardware_config(uint32_t, uint32_t, uint32_t, uint32_t);

                /* Python API - Set potential data source configuration for GPU compute
                 * task. */
                TaskConfigurator& set_morse_potential(double, double, double, double);

                /* Python API - Set algorithm configuration for a GPU compute task. */
                TaskConfigurator&
                set_vibwa_algorithm(double, double, uint32_t, uint32_t);

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
