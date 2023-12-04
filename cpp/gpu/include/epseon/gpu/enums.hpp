#pragma once

#include "epseon/libepseon.hpp"

#include <cassert>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>

#define PrecisionTypeAssertValueCount(count)                           \
    static_assert(                                                     \
        static_cast<int>(epseon::gpu::cpp::PrecisionType::_Last) == 2, \
        "The number of PrecisionTypes has changed."                    \
    );

namespace epseon::gpu::cpp {

    enum class PrecisionType {
        Float32,
        Float64,
        // If it is necessary to add new value, add it here, before _Last.
        _Last // Marker for last enum value.
    };

    class InvalidPrecisionTypeString : public std::exception {
      private:
        std::string message;

      public:
        InvalidPrecisionTypeString(std::string_view);
        const char* what() const noexcept override;
    };

    std::string   toString(PrecisionType);
    PrecisionType toPrecisionType(std::string_view prec);

    template <typename FP>
    PrecisionType getPrecisionType() {
        PrecisionTypeAssertValueCount(2);
        assert(false); // See template specializations in `enums.cpp`.
    };
} // namespace epseon::gpu::cpp
