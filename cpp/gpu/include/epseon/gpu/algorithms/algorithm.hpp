#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/device_interface.hpp"
#include <memory>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class Algorithm : public std::enable_shared_from_this<Algorithm<FP>> {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              public: /* Public destructor. */
                virtual ~Algorithm() {}

              public: /* Public methods. */
                virtual void run(std::stop_token, TaskHandle<FP>*) = 0;
            };

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
