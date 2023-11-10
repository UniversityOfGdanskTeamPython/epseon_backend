#pragma once

#define PY_SSIZE_T_CLEAN

#if defined(_DEBUG) && \
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

void hello();
