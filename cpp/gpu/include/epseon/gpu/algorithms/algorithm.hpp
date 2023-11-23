#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/device_interface.hpp"
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

              public:
                virtual void run(std::shared_ptr<TaskHandle<FP>>) = 0;
            };

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
