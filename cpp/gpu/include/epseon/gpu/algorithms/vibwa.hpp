#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/algorithms/algorithm.hpp"
#include "epseon/gpu/task_handle.hpp"
#include <iostream>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            class VibwaAlgorithm : public Algorithm<FP> {
                static_assert(
                    std::is_floating_point<FP>::value,
                    "FP must be an floating-point type."
                );

              public:
                VibwaAlgorithm() :
                    Algorithm<FP>() {}

                virtual void run(std::shared_ptr<TaskHandle<FP>> handle) {
                    std::cout << "Running" << std::endl;
                }
            };

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
