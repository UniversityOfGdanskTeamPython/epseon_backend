#pragma once

#define PY_SSIZE_T_CLEAN

#if defined(_DEBUG) &&                                                                 \
    (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__))
// Workaround for a VS 2022 issue.
// NOTE: This workaround knowingly violates the Python.h include order requirement:
// https://docs.python.org/3/c-api/intro.html#include-files
// See https://github.com/pybind/pybind11/pull/3497 for full context.
    #include <yvals.h>
    #if _MSVC_STL_VERSION >= 143
        #include <crtdefs.h>
    #endif
    #undef _DEBUG
    #include <Python.h>
    #define _DEBUG
#else
    #include <Python.h>
#endif

#include "epseon/libepseon.hpp"

#include <iostream>

SHARED_EXPORT void hello();

// Function to be called from Python
static PyObject* greet(PyObject* self, PyObject* args) {
    return PyUnicode_FromString("Hello, World from C++!");
}

// Method definition object for this extension, these arguments mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features or restrictions of the method, such as
// METH_NOARGS ml_doc:  Points to the contents of the docstring
static PyMethodDef GreetMethods[] = {
    {"greet", greet, METH_NOARGS, "Greet the world."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for its method definitions
static struct PyModuleDef libepseon_gpu = {
    PyModuleDef_HEAD_INIT, "greet", /* name of module */
    NULL,                           /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module keeps state
           in global variables. */
    GreetMethods};

// Module initialization function
PyMODINIT_FUNC PyInit__libepseon_gpu(void) {
    return PyModule_Create(&libepseon_gpu);
}
