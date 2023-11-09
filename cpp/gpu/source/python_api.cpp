#include "epseon_gpu/python_api.hpp"

#include "epseon_gpu/common.hpp"
#include "epseon_gpu/vulkan_application.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace epseon {
    namespace gpu {
        namespace python {

            std::string get_vulkan_version() {
                auto opt_app = epseon::gpu::cpp::VulkanApplication::create();
                if (opt_app.has_value()) {
                    return opt_app->get()->getVulkanAPIVersion();
                } else {
                    throw std::exception();
                }
            }

            std::vector<cpp::PhysicalDeviceInfo> get_physical_device_info() {
                auto opt_app = epseon::gpu::cpp::VulkanApplication::create();
                if (opt_app.has_value()) {
                    return opt_app->get()->getPhysicalDevicesInfo();
                } else {
                    throw std::exception();
                }
            }

            PYBIND11_MODULE(_libepseon_gpu, m) {
                m.doc() = "Sub package for interacting with GPU compute "
                          "capabilities.";
                m.def(
                    "get_vulkan_version", &get_vulkan_version, "Get Vulkan API version."
                );
                py::class_<vk::PhysicalDeviceSparseProperties>(
                    m, "PhysicalDeviceSparseProperties"
                )
                    .doc() =
                    "Wrapper around vk::PhysicalDeviceSparseProperties object.";

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
                                    return "integrated_gpu";
                                case vk::PhysicalDeviceType::eDiscreteGpu:
                                    return "discrete_gpu";
                                case vk::PhysicalDeviceType::eVirtualGpu:
                                    return "virtual_gpu";
                                case vk::PhysicalDeviceType::eCpu:
                                    return "cpu";
                                case vk::PhysicalDeviceType::eOther:
                                    return "other";
                            }
                            throw std::exception();
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

                py::class_<cpp::PhysicalDeviceInfo>(m, "PhysicalDeviceInfo")
                    .def_property_readonly(
                        "device_properties",
                        [](const cpp::PhysicalDeviceInfo& info) {
                            return info.deviceProperties;
                        }
                    )
                    .doc() =
                    "Container for physical device info retrieved from Vulkan API.";

                m.def(
                    "get_physical_device_info",
                    &get_physical_device_info,
                    "Get information about available physical devices.",
                    py::return_value_policy::move
                );
            }

        } // namespace python
    }     // namespace gpu
} // namespace epseon
