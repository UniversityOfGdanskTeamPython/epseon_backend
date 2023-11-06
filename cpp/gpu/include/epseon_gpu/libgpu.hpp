#pragma once

#include "epseon/libepseon.hpp"
#include "pyerrors.h"
#include "spdlog/logger.h"
#include "spdlog/spdlog.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace epseon {
    namespace gpu {

        class VulkanApplication {
          private:
            std::shared_ptr<spdlog::logger>      logger           = {};
            std::unique_ptr<vk::raii::Context>   context          = {};
            std::unique_ptr<vk::ApplicationInfo> application_info = {};
            std::unique_ptr<vk::raii::Instance>  instance         = {};

          private:
            VulkanApplication(
                std::shared_ptr<spdlog::logger>      logger,
                std::unique_ptr<vk::raii::Context>   context,
                std::unique_ptr<vk::ApplicationInfo> application_info,
                std::unique_ptr<vk::raii::Instance>  instance
            );

          public:
            static std::optional<std::unique_ptr<VulkanApplication>>
            create(uint32_t version = VK_MAKE_API_VERSION(0, 0, 1, 0));
            friend std::optional<std::unique_ptr<VulkanApplication>> create(uint32_t);

          public:
            std::string getVulkanAPIVersion();
        };
    } // namespace gpu
} // namespace epseon

SHARED_EXPORT void hello();

// Function to be called from Python
static PyObject* get_vulkan_version(PyObject* self, PyObject* args) {
    auto opt_app = epseon::gpu::VulkanApplication::create();
    if (opt_app.has_value()) {
        auto app = opt_app->get();
        return PyUnicode_FromString(app->getVulkanAPIVersion().c_str());
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Vulkan interface.");
        return NULL;
    }
}

// Method definition object for this extension, these arguments mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features or restrictions of the method, such as
// METH_NOARGS ml_doc:  Points to the contents of the docstring
static PyMethodDef LibGPUMethods[] = {
    {"get_vulkan_version", get_vulkan_version, METH_NOARGS, "Get Vulkan API version."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for its method definitions
static struct PyModuleDef libepseon_gpu = {
    PyModuleDef_HEAD_INIT,
    "_libepseon_gpu", /* name of module */
    NULL,             /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module keeps state
           in global variables. */
    LibGPUMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit__libepseon_gpu(void);
