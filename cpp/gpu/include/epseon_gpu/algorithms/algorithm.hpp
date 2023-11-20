#pragma once

#include "epseon_gpu/device_interface.hpp"
#include <memory>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class Algorithm {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              protected:
                std::shared_ptr<ComputeDeviceInterface> computeDeviceInterface = {};

              protected:
                Algorithm(
                    std::shared_ptr<ComputeDeviceInterface> computeDeviceInterface_
                ) :
                    computeDeviceInterface(computeDeviceInterface_) {}

              public:
                Algorithm()            = delete;
                Algorithm(Algorithm&)  = delete;
                Algorithm(Algorithm&&) = delete;
            };

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
