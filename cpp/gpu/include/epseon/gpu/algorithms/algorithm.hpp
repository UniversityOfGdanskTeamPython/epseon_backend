#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/device_interface.hpp"
#include <memory>

namespace epseon::gpu::cpp {

    template <typename FP>
    class Algorithm : public std::enable_shared_from_this<Algorithm<FP>> {
        static_assert(std::is_floating_point<FP>::value, "FP must be an floating-point type.");

      public: /* Public constructors. */
        // Default constructor
        Algorithm() = default;

        // Copy constructor
        Algorithm(const Algorithm&) = default;

        // Copy assignment operator
        Algorithm& operator=(const Algorithm&) = default;

        // Move constructor
        Algorithm(Algorithm&&) noexcept = default;

        // Move assignment operator
        Algorithm& operator=(Algorithm&&) noexcept = default;

      public: /* Public destructor. */
        virtual ~Algorithm() = default;

      public: /* Public methods. */
        virtual void run(const std::stop_token&, TaskHandle<FP>*) = 0;
    };

} // namespace epseon::gpu::cpp
