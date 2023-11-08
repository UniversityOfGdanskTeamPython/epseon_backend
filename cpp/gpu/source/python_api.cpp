#include "epseon_gpu/python_api.hpp"

#include "Python.h"
#include "descrobject.h"
#include "epseon_gpu/vulkan_application.hpp"
#include <cassert>
#include <cstddef>

namespace epseon {
    namespace gpu {
        namespace python {
            template <typename T>
            PyObject* std_vector_to_py_list(
                const std::vector<T>& cpp_vec, PyObject* (*ConvertToPy)(const T&)
            ) {
                PyObject* r = PyList_New(cpp_vec.size());
                if (!r) {
                    goto except;
                }
                for (Py_ssize_t i = 0; i < cpp_vec.size(); ++i) {
                    PyObject* item = (*ConvertToPy)(cpp_vec[i]);
                    if (!item || PyErr_Occurred() || PyList_SetItem(r, i, item)) {
                        goto except;
                    }
                }
                assert(!PyErr_Occurred());
                assert(r);
                goto finally;
            except:
                assert(PyErr_Occurred());
                // Clean up list
                if (r) {
                    // No PyList_Clear().
                    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(r); ++i) {
                        Py_XDECREF(PyList_GET_ITEM(r, i));
                    }
                    Py_DECREF(r);
                    r = NULL;
                }
            finally:
                return r;
            }

            extern "C" {

                // #define PyPhysicalDeviceProperties_member(name)                           \
//     {                                                                     \
//         #name, Py_T_OBJECT_EX, offsetof(PyPhysicalDeviceProperties, name) \
//     }
                //                 static PyMemberDef
                //                 PyPhysicalDeviceProperties_members[]{
                //                     PyPhysicalDeviceProperties_member(apiVersion),
                //                     PyPhysicalDeviceProperties_member(driverVersion),
                //                     PyPhysicalDeviceProperties_member(vendorID),
                //                     PyPhysicalDeviceProperties_member(deviceID),
                //                     PyPhysicalDeviceProperties_member(deviceType),
                //                     PyPhysicalDeviceProperties_member(deviceName),
                //                 };

                //                 static PyTypeObject CustomType = {
                //                     .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name
                //                     =
                //                         "_libepseon_gpu.PyPhysicalDeviceProperties",
                //                     .tp_doc = PyDoc_STR("Container for physical
                //                     device properties."), .tp_basicsize =
                //                     sizeof(PyPhysicalDeviceProperties), .tp_itemsize
                //                     = 0, .tp_flags = Py_TPFLAGS_DEFAULT |
                //                     Py_TPFLAGS_DISALLOW_INSTANTIATION |
                //                                 Py_TPFLAGS_IMMUTABLETYPE,
                //                     .tp_new     = PyPhysicalDeviceProperties_new,
                //                     .tp_dealloc =
                //                     (destructor)PyPhysicalDeviceProperties_dealloc,
                //                     .tp_members = PyPhysicalDeviceProperties_members,
                //                 };

                // static PyMemberDef PyPhysicalDeviceInfo_members[] = {
                //     {"device_properties",
                //      Py_T_OBJECT_EX,
                //      offsetof(PyPhysicalDeviceInfo, deviceProperties)},
                // };

                PyObject* get_vulkan_version(PyObject* self, PyObject* args) {
                    auto opt_app = epseon::gpu::cpp::VulkanApplication::create();
                    if (opt_app.has_value()) {
                        auto app = opt_app->get();
                        return PyUnicode_FromString(app->getVulkanAPIVersion().c_str());
                    } else {
                        PyErr_SetString(
                            PyExc_RuntimeError, "Failed to initialize Vulkan interface."
                        );
                        return NULL;
                    }
                }

                PyObject* get_physical_devices_info(PyObject* self, PyObject* args) {
                    auto opt_app = epseon::gpu::cpp::VulkanApplication::create();
                    if (opt_app.has_value()) {
                        auto        app     = opt_app->get();
                        auto        devices = app->getPhysicalDevicesInfo();
                        std::string retval  = {};

                        if (!devices.empty()) {
                            auto deviceName = devices[0].deviceProperties.deviceName;
                            retval          = {deviceName.begin(), deviceName.end()};
                        }

                        return PyUnicode_FromString(retval.c_str());
                    } else {
                        PyErr_SetString(
                            PyExc_RuntimeError, "Failed to initialize Vulkan interface."
                        );
                        return NULL;
                    }
                }

                // Method definition object for this extension, these arguments mean:
                // ml_name: The name of the method
                // ml_meth: Function pointer to the method implementation
                // ml_flags: Flags indicating special features or restrictions of the
                // method, such as ml_doc:  Points to the contents of the docstring
                static PyMethodDef LibGPUMethods[] = {
                    {"get_vulkan_version",
                     get_vulkan_version,
                     METH_NOARGS,
                     "Get Vulkan API version."},
                    {"get_physical_devices_info",
                     get_physical_devices_info,
                     METH_NOARGS,
                     "Get physical device info."},
                    {NULL, NULL, 0, NULL} /* Sentinel */
                };

                // Module definition
                // The arguments of this structure tell Python what to call your
                // extension, what it's methods are and where to look for its method
                // definitions
                static struct PyModuleDef libepseon_gpu = {
                    PyModuleDef_HEAD_INIT,
                    "_libepseon_gpu",
                    "Sub package for interacting with GPU compute capabilities.",
                    -1, /* size of per-interpreter state of the module, or -1 if the
                           module keeps state in global variables. */
                    LibGPUMethods
                };

                PyMODINIT_FUNC PyInit__libepseon_gpu(void) {
                    return PyModule_Create(&libepseon_gpu);
                }
            }
        } // namespace python
    }     // namespace gpu
} // namespace epseon
