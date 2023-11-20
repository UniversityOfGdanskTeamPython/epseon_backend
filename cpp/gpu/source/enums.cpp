#include "epseon_gpu/enums.hpp"
#include "fmt/format.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <string_view>

namespace epseon {
    namespace gpu {
        namespace cpp {

            InvalidPrecisionTypeString::InvalidPrecisionTypeString(std::string_view sv
            ) :
                message(
                    fmt::format("Invalid PrecisionType literal in string: \"{}\"", sv)
                ) {}

            const char* InvalidPrecisionTypeString::what() const noexcept {
                return this->message.c_str();
            };

            std::string toString(PrecisionType prec) {

                PrecisionTypeAssertValueCount(2);
                switch (prec) {
                    using enum PrecisionType;
                    case Float32:
                        return "Float32";
                    case Float64:
                        return "Float64";
                    default:
                        throw std::runtime_error("Unreachable");
                }
            }

            PrecisionType toPrecisionType(std::string_view precision) {
                std::string precision_lower_case(precision.begin(), precision.end());
                std::transform(
                    precision.begin(),
                    precision.end(),
                    precision_lower_case.begin(),
                    [](unsigned char c) {
                        return std::tolower(c);
                    }
                );

                PrecisionTypeAssertValueCount(2);
                if (precision_lower_case == "float32") {
                    return PrecisionType::Float32;
                } else if (precision_lower_case == "float64") {
                    return PrecisionType::Float64;
                } else {
                    throw InvalidPrecisionTypeString(precision_lower_case);
                }
            }

            template <>
            PrecisionType getPrecisionType<float>() {
                PrecisionTypeAssertValueCount(2);
                return PrecisionType::Float32;
            }

            template <>
            PrecisionType getPrecisionType<double>() {
                PrecisionTypeAssertValueCount(2);
                return PrecisionType::Float64;
            }
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
