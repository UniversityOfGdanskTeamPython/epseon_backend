#pragma once

#include "epseon_gpu/device_interface.hpp"
#include "epseon_gpu/task_configurator.hpp"
#include <memory>

namespace epseon {
    namespace gpu {
        namespace cpp {
            template <typename FP>
            class TaskHandle {
              private:
                std::shared_ptr<ComputeDeviceInterface> device;
                std::shared_ptr<TaskConfigurator<FP>>   config;

              public:
                TaskHandle(
                    std::shared_ptr<ComputeDeviceInterface> device_,
                    std::shared_ptr<TaskConfigurator<FP>>   config_
                ) :
                    device(device_),
                    config(config_) {}
            };

            template class TaskHandle<float>;
            template class TaskHandle<double>;

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
