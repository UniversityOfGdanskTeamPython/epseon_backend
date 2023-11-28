#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/algorithms/algorithm.hpp"
#include "epseon/gpu/task_handle.hpp"
#include "spdlog/fmt/bundled/core.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class VibwaAlgorithm : public Algorithm<FP> {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              public: /* Public constructors. */
                VibwaAlgorithm() :
                    Algorithm<FP>() {}

              public: /* Public destructor. */
                virtual ~VibwaAlgorithm() {}

              public: /* Public methods. */
                virtual void run(std::stop_token stop_token, TaskHandle<FP>* handle) {
                    {
                        if (stop_token.stop_requested()) {
                            return;
                        }
                        auto physical_device_ptr = handle->device->getPhysicalDevice();

                        for (auto        i = 0;
                             const auto& queue :
                             physical_device_ptr->getQueueFamilyProperties()) {
                            if (stop_token.stop_requested()) {
                                return;
                            }
                            // clang-format off
                            std::cout << fmt::format("queue_family_index #{}", i) << std::endl;
                            std::cout << fmt::format("queueCount {}", queue.queueCount) << std::endl;
                            std::cout << fmt::format("eCompute {}", bool(queue.queueFlags & vk::QueueFlagBits::eCompute)) << std::endl;
                            std::cout << fmt::format("eGraphics {}", bool(queue.queueFlags & vk::QueueFlagBits::eGraphics)) << std::endl;
                            std::cout << fmt::format("eProtected {}", bool(queue.queueFlags & vk::QueueFlagBits::eProtected)) << std::endl;
                            std::cout << fmt::format("eSparseBinding {}", bool(queue.queueFlags & vk::QueueFlagBits::eSparseBinding)) << std::endl;
                            std::cout << fmt::format("eTransfer {}", bool(queue.queueFlags & vk::QueueFlagBits::eTransfer)) << std::endl;
                            std::cout << std::endl;
                            // clang-format on
                            i++;
                        }

                        // std::vector<vk::DeviceQueueCreateInfo> queue_create_info;
                        // queue_create_info.emplace_back(
                        //     vk::DeviceQueueCreateFlags{},
                        //     queue_family_index,
                        //     1,
                        //     &queue_priority
                        // );
                        // auto queues =
                        //     physicalDevice.createDevice(vk::DeviceCreateInfo());
                        // auto logicalDevice =
                        // physicalDevice->createDevice(vk::DeviceCreateInfo().set);
                    }
                    std::cout << "Thread finished" << std::endl;
                }
            };

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
