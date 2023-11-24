#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/device_interface.hpp"
#include "epseon/gpu/task_configurator/task_configurator.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <thread>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class TaskHandle : public std::enable_shared_from_this<TaskHandle<FP>> {
              private:
                std::shared_ptr<ComputeDeviceInterface> device         = {};
                std::shared_ptr<TaskConfigurator<FP>>   config         = {};
                std::shared_ptr<std::jthread>           worker         = {};
                // Lock for guarding writes to worker pointer.
                // Currently this lock is not necessary as only one write to worker
                // pointer happens
                std::shared_ptr<std::atomic<bool>>      is_worker_done = {};

              public: /* Public constructors. */
                TaskHandle(
                    std::shared_ptr<ComputeDeviceInterface> device_,
                    std::shared_ptr<TaskConfigurator<FP>>   config_
                ) :
                    device(device_),
                    config(config_),
                    worker(nullptr),
                    is_worker_done(std::make_shared<std::atomic<bool>>(false)) {}

              protected: /* Protected methods. */
                void set_done_flag() {
                    /* Release all previous writes. */
                    // Nothing that was before the store can be observed after the
                    // operation. Things which were done after can be observed before.
                    // Refer to: https://www.youtube.com/watch?v=ZQFzMfHIxng
                    // CppCon 2017: Fedor Pikus "C++ atomics, from basic to advanced.
                    // What do they really do?"
                    this->is_worker_done->store(true, std::memory_order_release);
                }

                friend VibwaAlgorithm<FP>;

              public: /* Public methods. */
                void start_worker() {

                    if (this->worker) {
                        throw std::runtime_error(
                            "One worker is already running, can't start another one."
                        );
                    }
                    auto shared_this = this->shared_from_this();

                    this->worker = std::make_shared<std::jthread>(
                        [this, shared_this](std::stop_token stop_token) {
                            this->run(stop_token, shared_this);
                        }
                    );
                }

                void
                run(std::stop_token                 stop_token,
                    std::shared_ptr<TaskHandle<FP>> shared_this) {
                    const auto config         = this->config->getAlgorithmConfig();
                    const auto implementation = config->getImplementation();
                    implementation->run(stop_token, shared_this);
                }

                bool is_done() {
                    // Nothing that was after the load can move in front of it.
                    // Anything that was before can move after.
                    // Thus when paired whin store-release this should result in all
                    // reads after this load-acquire should see results of writes
                    // scheduled before store-release.
                    return is_worker_done->load(std::memory_order_acquire);
                }

                bool cancel() {

                    // Theoretically worker thread can be a nullptr.
                    if (this->worker) {
                        return this->worker->request_stop();
                    }
                    throw std::runtime_error(
                        "Calling wait() on non-existent worker is not allowed."
                    );
                }

                void wait() {

                    // Theoretically worker thread can be a nullptr.
                    if (this->worker) {
                        return this->worker->join();
                    }
                    throw std::runtime_error(
                        "Calling wait() on non-existent worker is not allowed."
                    );
                }
            };

            template class TaskHandle<float>;
            template class TaskHandle<double>;

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
