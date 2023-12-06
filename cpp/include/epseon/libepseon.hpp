#pragma once
#include <cassert>
#include <csignal>
#include <exception>
#include <iostream>
#include <string_view>

#include "fmt/format.h"

#if (defined(DEBUG) || (!defined(NDEBUG)) || defined(_DEBUG))
    #pragma message("[INFO] Compiling libepseon in debug mode.")
    #define LIB_EPSEON_DEBUG 1
    #define LIB_EPSEON_RELEASE 0
#else
    #pragma message("[INFO] Compiling libepseon in release mode.")
    #define LIB_EPSEON_DEBUG 0
    #define LIB_EPSEON_RELEASE 1
#endif

static_assert(
    LIB_EPSEON_DEBUG != 1 || LIB_EPSEON_RELEASE != 1,
    "LIB_EPSEON_DEBUG and LIB_EPSEON_RELEASE can't be both false."
);
static_assert(
    LIB_EPSEON_DEBUG != 0 || LIB_EPSEON_RELEASE != 0,
    "LIB_EPSEON_DEBUG and LIB_EPSEON_RELEASE can't be both true."
);
static_assert(
    LIB_EPSEON_DEBUG != 0 || LIB_EPSEON_DEBUG != 1, "LIB_EPSEON_DEBUG must be either 0 or 1."
);
static_assert(
    LIB_EPSEON_RELEASE != 0 || LIB_EPSEON_RELEASE != 1, "LIB_EPSEON_RELEASE must be either 0 or 1."
);

#if (LIB_EPSEON_DEBUG)
    #define LIB_EPSEON_ASSERT_TRUE(EXPRESSION) /* NOLINT: cppcoreguidelines-macro-usage */ \
        assert(EXPRESSION)
    #define LIB_EPSEON_ASSERT_FALSE(EXPRESSION) /* NOLINT: cppcoreguidelines-macro-usage */ \
        assert(!(static_cast<bool>(EXPRESSION)))

#else
    #define LIB_EPSEON_ASSERT_TRUE
#endif
