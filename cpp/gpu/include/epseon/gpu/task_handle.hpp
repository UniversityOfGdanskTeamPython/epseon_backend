#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/device_interface.hpp"
#include "epseon/gpu/task_configurator/task_configurator.hpp"
#include <memory>
#include <stdexcept>
#include <thread>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class TaskHandle {
              private:
                std::shared_ptr<ComputeDeviceInterface> device = {};
                std::shared_ptr<TaskConfigurator<FP>>   config = {};
                std::shared_ptr<std::jthread>           worker = {};

              public: /* Public constructors. */
                TaskHandle(
                    std::shared_ptr<ComputeDeviceInterface> device_,
                    std::shared_ptr<TaskConfigurator<FP>>   config_
                ) :
                    device(device_),
                    config(config_) {
                    this->start_worker();
                }

              public: /* Public methods. */
                void start_worker() {
                    if (worker) {
                        throw std::runtime_error(
                            "One worker is already running, can't start another one."
                        );
                    }
                    this->worker = std::make_shared<std::jthread>([this] {
                        this->run();
                    });
                }

                void run() {
                    auto& config = this->config->getAlgorithmConfig();
                }
            };

            template class TaskHandle<float>;
            template class TaskHandle<double>;

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
