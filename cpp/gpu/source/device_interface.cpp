#include "epseon_gpu/device_interface.hpp"
#include <vulkan/vulkan_raii.hpp>

namespace epseon {
    namespace gpu {
        namespace cpp {
            ComputeDeviceInterface::ComputeDeviceInterface(
                std::shared_ptr<ComputeContextState> computeContextState_,
                vk::raii::PhysicalDevice             physicalDevice_
            ) :
                computeContextState(computeContextState_),
                physicalDevice(physicalDevice_) {}

            std::shared_ptr<TaskConfiguratorVariant>
            ComputeDeviceInterface::getTaskConfigurator(PrecisionType prec) {
                switch (prec) {
                    case PrecisionType::Float32:
                        return std::make_shared<cpp::TaskConfiguratorVariant>(
                            cpp::TaskConfigurator<float>()
                        );
                    case PrecisionType::Float64:
                        return std::make_shared<cpp::TaskConfiguratorVariant>(
                            cpp::TaskConfigurator<double>()
                        );
                    default:
                        throw std::runtime_error(fmt::format(
                            "Unexpected PrecisionType enum value: {}",
                            static_cast<int64_t>(prec)
                        ));
                }
            }

            void submitTask() {}
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
