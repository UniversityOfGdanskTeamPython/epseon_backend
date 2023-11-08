#pragma once

#include "epseon/libepseon.hpp"

#include "Python.h"
#include "object.h"
#include "pytypedefs.h"

namespace epseon {
    namespace gpu {
        namespace python {
            extern "C" {

                /* Python API function - returns Vulkan version extracted from
                 * VkInstance. */
                PyObject* get_vulkan_version(PyObject* self, PyObject* args);

                /* Python API function - returns structured information about available
                 * physical devices.
                 */
                PyObject* get_physical_devices_info(PyObject* self, PyObject* args);

                // typedef struct {
                //     PyObject_HEAD;
                //     PyObject* apiVersion;
                //     PyObject* driverVersion;
                //     PyObject* vendorID;
                //     PyObject* deviceID;
                //     PyObject* deviceType;
                //     PyObject* deviceName;
                // } PyPhysicalDeviceProperties;

                // typedef struct {
                //     PyObject_HEAD;
                //     PyObject* deviceProperties;
                // } PyPhysicalDeviceInfo;

                // Module initialization function
                PyMODINIT_FUNC PyInit__libepseon_gpu(void);
            }
        } // namespace python
    }     // namespace gpu
} // namespace epseon
