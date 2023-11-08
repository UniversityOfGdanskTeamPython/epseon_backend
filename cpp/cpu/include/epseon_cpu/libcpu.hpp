#pragma once
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
static PyMethodDef LibCPUMethods[] = {
    {"greet", greet, METH_NOARGS, "Greet the world."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for its method definitions
static struct PyModuleDef libepseon_cpu = {
    PyModuleDef_HEAD_INIT,
    "_libepseon_cpu", /* name of module */
    NULL,             /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module keeps state
           in global variables. */
    LibCPUMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit__libepseon_cpu(void) {
    return PyModule_Create(&libepseon_cpu);
}
