#include "epseon_gpu/python/api.hpp"

#include "epseon_gpu/common.hpp"
#include "epseon_gpu/vulkan_application.hpp"
#include "pybind11/detail/common.h"
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_enums.hpp"
#include "vulkan/vulkan_structs.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace epseon {
    namespace gpu {
        namespace python {

            EpseonComputeContext::EpseonComputeContext(
                std::unique_ptr<cpp::VulkanApplication> application
            ) :
                application(std::move(application)) {}

            std::unique_ptr<EpseonComputeContext> EpseonComputeContext::create() {
                auto application = cpp::VulkanApplication::create();
                if (application) {
                    return std::make_unique<EpseonComputeContext>(std::move(application)
                    );
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

            PYBIND11_MODULE(_libepseon_gpu, m) {
                m.doc() = "Sub package for interacting with GPU compute "
                          "capabilities.";

                py::class_<vk::PhysicalDeviceSparseProperties>(
                    m, "PhysicalDeviceSparseProperties"
                )
                    .doc() =
                    "Wrapper around vk::PhysicalDeviceSparseProperties object.";

                /* Physical device limits - mostly max counts of different resources. */
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

                /* Properties of physical device retrieved from Vulkan API. */
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

                /* Properties of physical device memory retrieved from Vulkan API. */
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

                /* Properties of physical device memory retrieved from Vulkan API. */
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

                py::class_<EpseonComputeContext>(m, "EpseonComputeContext")
                    .def(
                        "create",
                        &EpseonComputeContext::create,
                        "Create instance of VulkanContext object.",
                        py::return_value_policy::move
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
                    .doc() = "Vulkan interface handle.";
            }

        } // namespace python
    }     // namespace gpu
} // namespace epseon
