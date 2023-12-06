#include "epseon/gpu/enums.hpp"
#include "epseon/gpu/libgpu.hpp"
#include "epseon/gpu/task_configurator/algorithm_config.hpp"
#include "epseon/gpu/task_configurator/potential_source.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace epseon {
    namespace gpu {
        namespace cpp {
            class LibGPUTest : public ::testing::Test {};

            TEST_F(LibGPUTest, EnsureCompatibleGPUAvailable) {
                auto ctx = ComputeContext::create();

                auto device_info = ctx->getPhysicalDevicesInfo();

                ASSERT_FALSE(device_info.empty());
            }

            TEST_F(LibGPUTest, SelectAndGetDevice) {
                auto ctx = ComputeContext::create();

                auto device_info_vector = ctx->getPhysicalDevicesInfo();
                for (auto device_info : device_info_vector) {
                    auto device = ctx->getDeviceInterface(device_info.deviceProperties.deviceID);
                    ASSERT_TRUE(&device->getPhysicalDevice());
                }
            }

            TEST_F(LibGPUTest, ConfigureTaskOnFirstDevice) {
                auto ctx = ComputeContext::create();

                auto device_info_vector = ctx->getPhysicalDevicesInfo();
                auto first_device_info  = *(device_info_vector.begin());
                auto first_device =
                    ctx->getDeviceInterface(first_device_info.deviceProperties.deviceID);
                auto cfg = first_device->getTaskConfigurator<float>();
                cfg->setHardwareConfig(
                       std::make_shared<HardwareConfig<float>>(500, 100, 16 * 1024 * 1024)
                )
                    .setAlgorithmConfig(
                        std::make_shared<VibwaAlgorithmConfig<float>>(87.62, 87.62, 0.1, 0.1, 0, 0)
                    )
                    .setPotentialSource(std::make_shared<MorsePotentialGenerator<float>>(
                        std::vector<MorsePotentialConfig<float>>{
                            MorsePotentialConfig<float>(5500.0, 0.6, 10, 0.0, 10.0, 500)
                        }
                    ));

                ASSERT_TRUE(cfg->isConfigured());
            }

            TEST_F(LibGPUTest, RunTaskOnFirstDevice) {
                auto ctx = ComputeContext::create();

                auto device_info_vector = ctx->getPhysicalDevicesInfo();
                auto first_device_info  = device_info_vector[0];
                auto first_device =
                    ctx->getDeviceInterface(first_device_info.deviceProperties.deviceID);
                auto cfg = first_device->getTaskConfigurator<float>();
                cfg->setHardwareConfig(
                       std::make_shared<HardwareConfig<float>>(500, 100, 16 * 1024 * 1024)
                )
                    .setAlgorithmConfig(
                        std::make_shared<VibwaAlgorithmConfig<float>>(87.62, 87.62, 0.1, 0.1, 0, 0)
                    )
                    .setPotentialSource(std::make_shared<MorsePotentialGenerator<float>>(
                        std::vector<MorsePotentialConfig<float>>{
                            MorsePotentialConfig<float>(5500.0, 0.6, 10, 0.0, 10.0, 500)
                        }
                    ));

                auto handle = first_device->submitTask(cfg);
                handle->startWorker();
                handle->wait();
            }

            TEST_F(LibGPUTest, RunTaskOnFirstDeviceWithRefsOutOfScope) {
                auto prepare = []() {
                    auto ctx = ComputeContext::create();

                    auto device_info_vector = ctx->getPhysicalDevicesInfo();
                    auto first_device_info  = device_info_vector[0];
                    auto first_device =
                        ctx->getDeviceInterface(first_device_info.deviceProperties.deviceID);
                    auto cfg = first_device->getTaskConfigurator<float>();
                    cfg->setHardwareConfig(
                           std::make_shared<HardwareConfig<float>>(500, 100, 16 * 1024 * 1024)
                    )
                        .setAlgorithmConfig(std::make_shared<VibwaAlgorithmConfig<float>>(
                            87.62, 87.62, 0.1, 0.1, 0, 0
                        ))
                        .setPotentialSource(std::make_shared<MorsePotentialGenerator<float>>(
                            std::vector<MorsePotentialConfig<float>>{
                                MorsePotentialConfig<float>(5500.0, 0.6, 10, 0.0, 10.0, 500)
                            }
                        ));
                    return first_device->submitTask(cfg);
                };

                auto handle = prepare();
                handle->startWorker();
                handle->wait();
                ASSERT_TRUE(handle->isDone());
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
