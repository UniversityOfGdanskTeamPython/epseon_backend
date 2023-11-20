#pragma once
#include "epseon_gpu/algorithms/algorithm.hpp"

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class VibwaAlgorithms : public Algorithm<FP> {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              protected:
                TaskConfigurator<FP> configurator;

              public:
                VibwaAlgorithms(
                    std::shared_ptr<ComputeDeviceInterface> computeDeviceInterface_
                ) :
                    Algorithm<FP>(computeDeviceInterface_) {}
            };

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
