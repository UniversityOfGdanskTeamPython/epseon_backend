#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/device_interface.hpp"
#include "epseon/gpu/task_configurator/task_configurator.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <system_error>
#include <thread>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class TaskHandle : public std::enable_shared_from_this<TaskHandle<FP>> {
              private:
                std::shared_ptr<ComputeDeviceInterface> device            = {};
                std::shared_ptr<TaskConfigurator<FP>>   config            = {};
                std::atomic<bool>                       is_worker_done    = false;
                std::atomic<bool>                       is_worker_started = false;
                std::jthread                            worker            = {};

              public: /* Public constructors. */
                TaskHandle(
                    std::shared_ptr<ComputeDeviceInterface> device_,
                    std::shared_ptr<TaskConfigurator<FP>>   config_
                ) :
                    device(device_),
                    config(config_),
                    is_worker_done(false),
                    is_worker_started(false),
                    worker() {
                    /* We can't start worker in constructor as it takes a shared
                     * pointer to this handle object, which will not be initialized
                     * within constructor call. Therefore start_worker() must be
                     * called afterwards.
                     */
                }

                // Default constructor.
                TaskHandle() = default;

                // Copy constructor.
                TaskHandle(const TaskHandle&) = default;

                // Copy assignment operator.
                TaskHandle& operator=(const TaskHandle&) = default;

                // Move constructor.
                TaskHandle(TaskHandle&&) noexcept = default;

                // Move assignment operator.
                TaskHandle& operator=(TaskHandle&&) noexcept = default;

              public: /* Public destructor. */
                ~TaskHandle() {}

              protected: /* Protected methods. */
                void setDoneFlag() {
                    /* Release all previous writes. */
                    // Nothing that was before the store can be observed after the
                    // operation. Things which were done after can be observed before.
                    // Refer to: https://www.youtube.com/watch?v=ZQFzMfHIxng
                    // CppCon 2017: Fedor Pikus "C++ atomics, from basic to advanced.
                    // What do they really do?"
                    this->is_worker_done.store(true, std::memory_order_release);
                }

                void setStartedFlag() {
                    // See setDoneFlag for deeper explanation why
                    // std::memory_order_release is used here.
                    this->is_worker_started.store(true, std::memory_order_release);
                }

                void setNotDoneFlag() {
                    // See setDoneFlag for deeper explanation why
                    // std::memory_order_release is used here.
                    this->is_worker_done.store(false, std::memory_order_release);
                }

                void setNotStartedFlag() {
                    // See setDoneFlag for deeper explanation why
                    // std::memory_order_release is used here.
                    this->is_worker_started.store(false, std::memory_order_release);
                }

                friend VibwaAlgorithm<FP>;

              public: /* Public methods. */
                /* Create and start underlying worker thread.*/
                void startWorker() {
                    if (this->isRunning()) {
                        throw std::runtime_error(
                            "One worker is already running, can't start another one."
                        );
                    }
                    this->setNotDoneFlag();
                    this->setStartedFlag();
                    this->worker = std::jthread(this->run, this);
                }

                /* Code run withing worker thread. */
                void static run(std::stop_token stop_token, TaskHandle<FP>* this_ptr) {
                    const auto config         = this_ptr->config->getAlgorithmConfig();
                    const auto implementation = config->getImplementation();
                    implementation->run(stop_token, this_ptr);
                    this_ptr->setDoneFlag();
                    this_ptr->setNotStartedFlag();
                }

                /* Check if underlying worker thread finished its work.
                 * This doesn't check if thread even started, it will be false both if
                 * it is currently running and if it was never started. Use is_running()
                 * to clarify which of those is the case.
                 */
                bool isDone() const {
                    // Nothing that was after the load can move in front of it.
                    // Anything that was before can move after.
                    // Thus when paired whin store-release this should result in all
                    // reads after this load-acquire should see results of writes
                    // scheduled before store-release.
                    return is_worker_done.load(std::memory_order_acquire);
                }

                /* Check if underlying worker thread was ever started. */
                bool isStarted() const {
                    return is_worker_started.load(std::memory_order_acquire);
                }

                /* Check if underlying worker thread is currently running. */
                bool isRunning() const {
                    return isStarted() && !isDone();
                }

                bool cancel() {
                    // Theoretically worker thread can be a nullptr.
                    if (this->isRunning()) {
                        // This will only request a stop, but it is up to worker to
                        // check if stop was requested and whether to respond at all.
                        return this->worker.request_stop();
                    }
                    return false;
                }

                void wait() {
                    // Theoretically worker thread can be a nullptr.
                    if (this->isRunning()) {
                        this->worker.join();
                        return;
                    }
                }

              public: /* Public getters. */
                const TaskConfigurator<FP>& getTaskConfigurator() const {
                    return *this->config;
                }

                const ComputeDeviceInterface& getDeviceInterface() const {
                    return *this->device;
                }
            };

            template class TaskHandle<float>;
            template class TaskHandle<double>;

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
