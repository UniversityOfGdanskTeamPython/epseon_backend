#include "epseon/gpu/python/api.hpp"

#include "epseon/gpu/common.hpp"
#include "epseon/gpu/compute_context.hpp"
#include "epseon/gpu/enums.hpp"
#include "epseon/gpu/task_configurator/task_configurator.hpp"
#include "fmt/format.h"
#include "pybind11/detail/common.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_enums.hpp"
#include "vulkan/vulkan_structs.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace epseon {
    namespace gpu {
        namespace python {

            MorsePotentialConfig MorsePotentialConfig::create(
                double   dissociation_energy_,
                double   equilibrium_bond_distance_,
                double   well_width_,
                double   min_r_,
                double   max_r_,
                uint32_t point_count_
            ) {
                return {std::move(cpp::MorsePotentialConfig<double>(
                    dissociation_energy_,
                    equilibrium_bond_distance_,
                    well_width_,
                    min_r_,
                    max_r_,
                    point_count_
                ))};
            }

            const cpp::MorsePotentialConfig<double>&
            MorsePotentialConfig::getConfiguration() const {
                return this->configuration;
            }

            TaskConfigurator::TaskConfigurator(
                cpp::PrecisionType                            precision_,
                std::shared_ptr<cpp::TaskConfiguratorVariant> configurator_
            ) :
                precision(precision_),
                configurator(configurator_) {}

            // Copy constructor
            TaskConfigurator::TaskConfigurator(const TaskConfigurator& other) :
                precision(other.precision),
                configurator(other.configurator) {}

            // Move constructor
            TaskConfigurator::TaskConfigurator(TaskConfigurator&& other) noexcept :
                precision(std::move(other.precision)),
                configurator(std::move(other.configurator)) {}

            // Copy assignment operator
            TaskConfigurator& TaskConfigurator::operator=(const TaskConfigurator& other
            ) {
                if (this != &other) {
                    precision    = other.precision;
                    configurator = other.configurator;
                }
                return *this;
            }

            // Move assignment operator
            TaskConfigurator& TaskConfigurator::operator=(TaskConfigurator&& other
            ) noexcept {
                if (this != &other) {
                    precision    = std::move(other.precision);
                    configurator = std::move(other.configurator);
                }
                return *this;
            }

            TaskConfigurator& TaskConfigurator::set_hardware_config(
                uint32_t potential_buffer_size,
                uint32_t group_size,
                uint32_t allocation_block_size
            ) {
                auto helper = [potential_buffer_size,
                               group_size,
                               allocation_block_size,
                               this]<typename T>(T val_) {
                    auto& cfg =
                        std::get<cpp::TaskConfigurator<T>>(*(this->configurator));
                    cfg.setHardwareConfig(std::make_unique<cpp::HardwareConfig<T>>(
                        potential_buffer_size, group_size, allocation_block_size
                    ));
                };

                PrecisionTypeAssertValueCount(2);
                switch (this->precision) {
                    case cpp::PrecisionType::Float32: {
                        helper(float(1.0));
                        break;
                    }
                    case cpp::PrecisionType::Float64: {
                        helper(double(1.0));
                        break;
                    }
                    default:
                        throw py::value_error(fmt::format(
                            "Invalid PrecisionType value {}", toString(this->precision)
                        ));
                }
                return *this;
            }

            TaskConfigurator& TaskConfigurator::set_morse_potential(
                std::vector<MorsePotentialConfig> configurations
            ) {
                auto helper = [&configurations, this]<typename T>(T val_) {
                    auto& cfg =
                        std::get<cpp::TaskConfigurator<T>>(*(this->configurator));
                    std::optional<uint32_t> point_count = std::nullopt;

                    std::vector<cpp::MorsePotentialConfig<T>> configurations_cpp(
                        configurations.size()
                    );
                    for (const auto& element : configurations) {
                        auto current_element_point_count =
                            element.getConfiguration().getPointCount();
                        if (point_count.has_value() &&
                            point_count.value() != current_element_point_count) {
                            throw std::runtime_error(fmt::format(
                                "All Morse potentials must have same point "
                                "count, but previous ones had {} and current one has.",
                                point_count.value(),
                                current_element_point_count
                            ));
                        } else {
                            point_count = {current_element_point_count};
                        }
                        configurations_cpp.emplace_back(
                            element.getConfiguration().getDissociationEnergy<T>(),
                            element.getConfiguration().getEquilibriumBondDistance<T>(),
                            element.getConfiguration().getWellWidth<T>(),
                            element.getConfiguration().getMinR<T>(),
                            element.getConfiguration().getMaxR<T>(),
                            current_element_point_count
                        );
                    }

                    cfg.setPotentialSource(
                        std::make_unique<cpp::MorsePotentialGenerator<T>>(
                            std::move(configurations_cpp)
                        )
                    );
                };

                PrecisionTypeAssertValueCount(2);
                switch (this->precision) {
                    case cpp::PrecisionType::Float32: {
                        helper(float(1.0));
                        break;
                    }
                    case cpp::PrecisionType::Float64: {
                        helper(double(1.0));
                        break;
                    }
                    default:
                        throw py::value_error(fmt::format(
                            "Invalid PrecisionType value {}", toString(this->precision)
                        ));
                }
                return *this;
            }

            TaskConfigurator& TaskConfigurator::set_vibwa_algorithm(
                double   integration_step,
                double   min_distance_to_asymptote,
                uint32_t min_level,
                uint32_t max_level
            ) {
                auto helper = [integration_step,
                               min_distance_to_asymptote,
                               min_level,
                               max_level,
                               this]<typename T>(T val_) {
                    auto& cfg =
                        std::get<cpp::TaskConfigurator<T>>(*(this->configurator));
                    cfg.setAlgorithmConfig(
                        std::make_unique<cpp::VibwaAlgorithmConfig<T>>(
                            static_cast<T>(integration_step),
                            static_cast<T>(min_distance_to_asymptote),
                            min_level,
                            max_level
                        )
                    );
                };

                PrecisionTypeAssertValueCount(2);
                switch (this->precision) {
                    case cpp::PrecisionType::Float32: {
                        helper(float(1.0));
                        break;
                    }
                    case cpp::PrecisionType::Float64: {
                        helper(double(1.0));
                        break;
                    }
                    default:
                        throw py::value_error(fmt::format(
                            "Invalid PrecisionType value {}", toString(this->precision)
                        ));
                }
                return *this;
            }

            bool TaskConfigurator::is_configured() const {
                auto helper = [this]<typename T>(T val_) {
                    auto& cfg =
                        std::get<cpp::TaskConfigurator<T>>(*(this->configurator));
                    return cfg.isConfigured();
                };

                PrecisionTypeAssertValueCount(2);
                switch (this->precision) {
                    case cpp::PrecisionType::Float32: {
                        return helper(float(1.0));
                    }
                    case cpp::PrecisionType::Float64: {
                        return helper(double(1.0));
                    }
                    default:
                        throw py::value_error(fmt::format(
                            "Invalid PrecisionType value {}", toString(this->precision)
                        ));
                }
            }

            ComputeDeviceInterface::ComputeDeviceInterface(
                std::shared_ptr<cpp::ComputeDeviceInterface> device_
            ) :
                device(device_) {}

            TaskConfigurator
            ComputeDeviceInterface::get_task_configurator(const std::string precision) {
                auto precision_enum_value = [&precision]() {
                    try {
                        return cpp::toPrecisionType(precision);
                    } catch (cpp::InvalidPrecisionTypeString e) {
                        throw py::value_error(e.what());
                    }
                }();

                PrecisionTypeAssertValueCount(2);
                switch (precision_enum_value) {
                    case cpp::PrecisionType::Float32:
                        return TaskConfigurator{
                            precision_enum_value,
                            device->getTaskConfigurator(precision_enum_value)
                        };
                    case cpp::PrecisionType::Float64:
                        return TaskConfigurator{
                            precision_enum_value,
                            device->getTaskConfigurator(precision_enum_value)
                        };
                    default:
                        throw std::runtime_error("Unreachable.");
                }
                assert(false);
            }

            TaskHandleVariant
            ComputeDeviceInterface::submit_task(const TaskConfigurator& task_config) {
                if (!task_config.is_configured()) {
                    throw std::runtime_error("TaskConfigurator submitted for execution "
                                             "before fully configured.");
                }
                auto helper = [task_config,
                               this]<typename T>(T val_) -> TaskHandleVariant {
                    auto& config =
                        std::get<cpp::TaskConfigurator<T>>(*(task_config.configurator));
                    auto new_shared_config_clone =
                        std::make_shared<cpp::TaskConfigurator<T>>(config);
                    auto task_handle =
                        this->device->submitTask(new_shared_config_clone);

                    return {task_handle};
                };

                PrecisionTypeAssertValueCount(2);
                switch (task_config.precision) {
                    case cpp::PrecisionType::Float32: {
                        return helper(float(1.0));
                    }
                    case cpp::PrecisionType::Float64: {
                        return helper(double(1.0));
                    }
                    default:
                        throw py::value_error(fmt::format(
                            "Invalid PrecisionType value {}",
                            toString(task_config.precision)
                        ));
                }
            }

            EpseonComputeContext::EpseonComputeContext(
                std::shared_ptr<cpp::ComputeContext> application_
            ) :
                application(application_) {}

            EpseonComputeContext EpseonComputeContext::create() {
                auto application = cpp::ComputeContext::create();
                if (application) {
                    return EpseonComputeContext{application};
                } else {
                    throw std::runtime_error("Failed to create EpseonComputeContext.");
                }
            };

            std::string EpseonComputeContext::get_vulkan_version() {
                return application->getVulkanAPIVersion();
            }

            std::vector<cpp::PhysicalDeviceInfo>
            EpseonComputeContext::get_physical_device_info() {
                return application->getPhysicalDevicesInfo();
            }

            ComputeDeviceInterface
            EpseonComputeContext::get_device_interface(uint32_t device_id) {
                return {application->getDeviceInterface(device_id)};
            }

            PYBIND11_MODULE(_libepseon_gpu, m) {
                m.doc() = "Sub package for interacting with GPU compute "
                          "capabilities.";

                py::class_<vk::PhysicalDeviceSparseProperties>(
                    m, "PhysicalDeviceSparseProperties"
                )
                    .doc() =
                    "Wrapper around vk::PhysicalDeviceSparseProperties object.";

                /* Python API -  Wrapper around container class for physical device
                 * limits - mostly max counts of different resources. */
                py::class_<vk::PhysicalDeviceLimits>(m, "PhysicalDeviceLimits")
                    .def_property_readonly(
                        "max_compute_shared_memory_size",
                        [](const vk::PhysicalDeviceLimits& props) {
                            return props.maxComputeSharedMemorySize;
                        }
                    )
                    .def_property_readonly(
                        "max_compute_work_group_count",
                        [](const vk::PhysicalDeviceLimits& props
                        ) -> std::tuple<uint32_t, uint32_t, uint32_t> {
                            return {
                                props.maxComputeWorkGroupCount[0],
                                props.maxComputeWorkGroupCount[1],
                                props.maxComputeWorkGroupCount[2]
                            };
                        }
                    )
                    .def_property_readonly(
                        "max_compute_work_group_invocations",
                        [](const vk::PhysicalDeviceLimits& props) {
                            return props.maxComputeWorkGroupInvocations;
                        }
                    )
                    .def_property_readonly(
                        "max_compute_work_group_size",
                        [](const vk::PhysicalDeviceLimits& props) {
                            return props.maxComputeSharedMemorySize;
                        }
                    )
                    .doc() = "Physical device limits - mostly max counts of different "
                             "resources.";

                /* Python API - Wrapper around container class for properties of
                 * physical device retrieved from Vulkan API. */
                py::class_<vk::PhysicalDeviceProperties>(m, "PhysicalDeviceProperties")
                    .def_property_readonly(
                        "api_version",
                        [](const vk::PhysicalDeviceProperties& props) {
                            return common::vulkan_version_to_string(props.apiVersion);
                        }
                    )
                    .def_property_readonly(
                        "driver_version",
                        [](const vk::PhysicalDeviceProperties& props) {
                            return common::vulkan_version_to_string(props.driverVersion
                            );
                        }
                    )
                    .def_property_readonly(
                        "vendor_id",
                        [](const vk::PhysicalDeviceProperties& props) {
                            return props.vendorID;
                        }
                    )
                    .def_property_readonly(
                        "device_id",
                        [](const vk::PhysicalDeviceProperties& props) {
                            return props.deviceID;
                        }
                    )
                    .def_property_readonly(
                        "device_type",
                        [](const vk::PhysicalDeviceProperties& props) {
                            switch (props.deviceType) {
                                case vk::PhysicalDeviceType::eIntegratedGpu:
                                    return "INTEGRATED_GPU";
                                case vk::PhysicalDeviceType::eDiscreteGpu:
                                    return "DISCRETE_GPU";
                                case vk::PhysicalDeviceType::eVirtualGpu:
                                    return "VIRTUAL_GPU";
                                case vk::PhysicalDeviceType::eCpu:
                                    return "CPU";
                                case vk::PhysicalDeviceType::eOther:
                                    return "OTHER";
                            }
                            throw std::runtime_error("Unknown physical device type.");
                        }
                    )
                    .def_property_readonly(
                        "device_name",
                        [](const vk::PhysicalDeviceProperties& props) {
                            return std::string(props.deviceName.data());
                        }
                    )
                    .def_property_readonly(
                        "pipeline_cache_uuid",
                        [](const vk::PhysicalDeviceProperties& props) {
                            return std::vector<uint8_t>(
                                props.pipelineCacheUUID.begin(),
                                props.pipelineCacheUUID.end()
                            );
                        }
                    )
                    .def_property_readonly(
                        "limits",
                        [](const vk::PhysicalDeviceProperties& props) {
                            return props.limits;
                        }
                    )
                    .def_property_readonly(
                        "sparse_properties",
                        [](const vk::PhysicalDeviceProperties& props) {
                            return props.sparseProperties;
                        }
                    )
                    .doc() = "Properties of physical device retrieved from Vulkan API.";

                /* Python API - Wrapper around container class for memory heap info. */
                py::class_<vk::MemoryHeap>(m, "MemoryHeap")
                    .def_property_readonly(
                        "size",
                        [](const vk::MemoryHeap& heap) {
                            return heap.size;
                        }
                    )
                    .def_property_readonly(
                        "flags",
                        [](const vk::MemoryHeap& heap) {
                            std::vector<std::string> flags = {};
                            if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal)
                                flags.push_back("DEVICE_LOCAL");
                            if (heap.flags & vk::MemoryHeapFlagBits::eMultiInstance)
                                flags.push_back("MULTI_INSTANCE");
                            if (heap.flags & vk::MemoryHeapFlagBits::eMultiInstanceKHR)
                                flags.push_back("MULTI_INSTANCE_KHR");
                            return flags;
                        }
                    )
                    .doc() = "Wrapper around vk::MemoryHeap object.";

                /* Python API - Wrapper around container class for memory type info. */
                py::class_<vk::MemoryType>(m, "MemoryType")
                    .def_property_readonly(
                        "heap_index",
                        [](const vk::MemoryType& mem_type) {
                            return mem_type.heapIndex;
                        }
                    )
                    .def_property_readonly(
                        "flags",
                        [](const vk::MemoryType& mem_type) {
                            std::vector<std::string> flags = {};
                            if (mem_type.propertyFlags &
                                vk::MemoryPropertyFlagBits::eDeviceLocal)
                                flags.push_back("DEVICE_LOCAL");
                            if (mem_type.propertyFlags &
                                vk::MemoryPropertyFlagBits::eHostVisible)
                                flags.push_back("HOST_VISIBLE");
                            if (mem_type.propertyFlags &
                                vk::MemoryPropertyFlagBits::eHostCoherent)
                                flags.push_back("HOST_COHERENT");
                            if (mem_type.propertyFlags &
                                vk::MemoryPropertyFlagBits::eHostCached)
                                flags.push_back("HOST_CACHED");
                            if (mem_type.propertyFlags &
                                vk::MemoryPropertyFlagBits::eLazilyAllocated)
                                flags.push_back("LAZILY_ALLOCATED");
                            if (mem_type.propertyFlags &
                                vk::MemoryPropertyFlagBits::eProtected)
                                flags.push_back("PROTECTED");
                            if (mem_type.propertyFlags &
                                vk::MemoryPropertyFlagBits::eDeviceCoherentAMD)
                                flags.push_back("DEVICE_COHERENT_AMD");
                            if (mem_type.propertyFlags &
                                vk::MemoryPropertyFlagBits::eDeviceUncachedAMD)
                                flags.push_back("DEVICE_UNCACHED_AMD");
                            if (mem_type.propertyFlags &
                                vk::MemoryPropertyFlagBits::eRdmaCapableNV)
                                flags.push_back("RDMA_CAPABLE_NV");
                            return flags;
                        }
                    )
                    .doc() = "Wrapper around vk::MemoryHeap object.";

                /* Python API - Wrapper around container class for properties of
                 * physical device memory retrieved from Vulkan API. */
                py::class_<vk::PhysicalDeviceMemoryProperties>(
                    m, "PhysicalDeviceMemoryProperties"
                )
                    .def_property_readonly(
                        "memory_heaps",
                        [](const vk::PhysicalDeviceMemoryProperties& props) {
                            std::vector<vk::MemoryHeap> memoryHeaps = {};
                            for (uint32_t i = 0; i < props.memoryHeapCount; i++) {
                                memoryHeaps.push_back(props.memoryHeaps[i]);
                            }
                            return memoryHeaps;
                        }
                    )
                    .def_property_readonly(
                        "memory_types",
                        [](const vk::PhysicalDeviceMemoryProperties& props) {
                            std::vector<vk::MemoryType> memoryTypes = {};
                            for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
                                memoryTypes.push_back(props.memoryTypes[i]);
                            }
                            return memoryTypes;
                        }
                    )
                    .doc() =
                    "Wrapper around vk::PhysicalDeviceMemoryProperties object.";

                /* Python API - Wrapper class around PhysicalDeviceInfo class. */
                py::class_<cpp::PhysicalDeviceInfo>(m, "PhysicalDeviceInfo")
                    .def_property_readonly(
                        "device_properties",
                        [](const cpp::PhysicalDeviceInfo& info) {
                            return info.deviceProperties;
                        }
                    )
                    .def_property_readonly(
                        "memory_properties",
                        [](const cpp::PhysicalDeviceInfo& info) {
                            return info.memoryProperties;
                        }
                    )
                    .doc() =
                    "Container for physical device info retrieved from Vulkan API.";

#define TaskHandlerPyInterface(cls)                                                    \
    /* ---------------------------------------------------------------------------- */ \
    py::class_<cls>(m, #cls)                                                           \
        .def(                                                                          \
            "get_status_message",                                                      \
            &cls::get_status_message,                                                  \
            "Get status message explaining current execution stage."                   \
        )                                                                              \
        .def("is_done", &cls::is_done, "Check if task already finished execution.")    \
        .def("wait", &cls::wait, "Block and wait for task to finish.")                 \
        .doc() = "Handle object for referencing double precision GPU compute task."

                TaskHandlerPyInterface(TaskHandleFloat32);
                TaskHandlerPyInterface(TaskHandleFloat64);

                py::class_<MorsePotentialConfig>(m, "MorsePotentialConfig")
                    .def(
                        py::init(&MorsePotentialConfig::create),
                        py::arg("dissociation_energy"),
                        py::arg("equilibrium_bond_distance"),
                        py::arg("well_width"),
                        py::arg("min_r"),
                        py::arg("max_r"),
                        py::arg("point_count"),
                        "Create instance of MorsePotentialConfig class."
                    )
                    .doc() = "Configuration of single Morse potential curve.";

                /* Python API - Wrapper class around TaskConfigurator class. */
                py::class_<TaskConfigurator>(m, "TaskConfigurator")
                    .def(
                        "set_hardware_config",
                        &TaskConfigurator::set_hardware_config,
                        py::arg("potential_buffer_size"),
                        py::arg("group_size"),
                        py::arg("allocation_block_size"),
                        "Set hardware configuration for a GPU compute task."
                    )
                    .def(
                        "set_morse_potential",
                        &TaskConfigurator::set_morse_potential,
                        py::arg("configurations"),
                        "Set potential data source configuration for GPU compute "
                        "task."
                    )
                    .def(
                        "set_vibwa_algorithm",
                        &TaskConfigurator::set_vibwa_algorithm,
                        py::arg("integration_step"),
                        py::arg("min_distance_to_asymptote"),
                        py::arg("min_level"),
                        py::arg("max_level"),
                        "Set algorithm configuration for a GPU compute task."
                    )
                    .def(
                        "is_configured",
                        &TaskConfigurator::is_configured,
                        "Check if this instance is fully configured, i.e. it has "
                        "been "
                        "assigned a valid hardware configuration, potential source "
                        "and "
                        "algorithm config."
                    )
                    .doc() = "Builder for configuring GPU compute task.";

                // Python API - Wrapper class for ComputeDeviceInterface class.
                py::class_<ComputeDeviceInterface>(m, "ComputeDeviceInterface")
                    .def(
                        "get_task_configurator",
                        &ComputeDeviceInterface::get_task_configurator,
                        "Get builder instance for configuring GPU compute task."
                    )
                    .def(
                        "submit_task",
                        &ComputeDeviceInterface::submit_task,
                        "Submit task for execution. Will raise RuntimeError upon "
                        "receiving not fully configured TaskConfigurator."
                    )
                    .doc() = "Interface to particular Vulkan device.";

                // Python API - Wrapper class for EpseonComputeContext class.
                py::class_<EpseonComputeContext>(m, "EpseonComputeContext")
                    .def(
                        "create",
                        &EpseonComputeContext::create,
                        "Create instance of VulkanContext object."
                    )
                    .def(
                        "get_vulkan_version",
                        &EpseonComputeContext::get_vulkan_version,
                        "Get Vulkan API version."
                    )
                    .def(
                        "get_physical_device_info",
                        &EpseonComputeContext::get_physical_device_info,
                        "Get information about available physical devices."
                    )
                    .def(
                        "get_device_interface",
                        &EpseonComputeContext::get_device_interface,
                        "Get interface for running algorithms on Vulkan devices."
                    )
                    .doc() = "Vulkan interface handle.";
            }

        } // namespace python
    }     // namespace gpu
} // namespace epseon
