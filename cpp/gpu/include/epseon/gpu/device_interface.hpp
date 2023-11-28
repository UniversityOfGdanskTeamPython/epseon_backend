#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/compute_context.hpp"
#include "epseon/gpu/task_configurator/task_configurator.hpp"
#include "epseon/gpu/task_handle.hpp"
#include "fmt/format.h"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace epseon {
    namespace gpu {
        namespace cpp {

            typedef std::variant<TaskConfigurator<float>, TaskConfigurator<double>>
                TaskConfiguratorVariant;

            struct BufferRequirements {
              public:
                uint64_t buffer_size_bytes = {};
                uint32_t binding           = {};
            };

            struct ShaderMemoryRequirements {
              public:
                std::vector<BufferRequirements> buffers;

              public:
                uint64_t get_total_size_bytes() const;
            };

            class ComputeDeviceInterface
                : public std::enable_shared_from_this<ComputeDeviceInterface> {
              private: /* Private members. */
                std::shared_ptr<ComputeContextState>      computeContextState;
                std::shared_ptr<vk::raii::PhysicalDevice> physicalDevice;

              public: /* Public constructors. */
                ComputeDeviceInterface(std::shared_ptr<ComputeContextState>, std::shared_ptr<vk::raii::PhysicalDevice>);

              public: /* Public destructor. */
                ~ComputeDeviceInterface() {}

              public: /* Public methods. */
                template <typename FP>
                std::shared_ptr<TaskConfigurator<FP>> getTaskConfigurator() {
                    return std::make_shared<TaskConfigurator<FP>>();
                }

                std::shared_ptr<vk::raii::PhysicalDevice> getPhysicalDevice();

                template <typename FP>
                // Namespaces specified explicitly to avoid confusion.
                std::shared_ptr<epseon::gpu::cpp::TaskHandle<FP>>
                submitTask(std::shared_ptr<TaskConfigurator<FP>> task_config) {
                    if (!task_config->isConfigured()) {
                        throw std::runtime_error(
                            "TaskConfigurator wasn't fully configured before "
                            "submitting for execution."
                        );
                    }
                    return std::make_shared<TaskHandle<FP>>(
                        this->shared_from_this(), task_config
                    );
                }
            };
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
