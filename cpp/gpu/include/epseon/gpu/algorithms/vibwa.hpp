#pragma once

#include "epseon/gpu/predecl.hpp"

#include "epseon/gpu/algorithms/algorithm.hpp"
#include "epseon/gpu/task_handle.hpp"
#include "spdlog/fmt/bundled/core.h"
#include <cstdint>
#include <iostream>
#include <unistd.h>

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

                virtual void
                run(std::stop_token                 stop_token,
                    std::shared_ptr<TaskHandle<FP>> handle) {
                    std::cout << "Running" << std::endl;

                    uint32_t acc = 0;
                    for (uint32_t i = 0; i < 4; i++) {
                        std::cout << fmt::format("Running +{}", acc) << std::endl;

                        if (stop_token.stop_requested())
                            break;

                        sleep(1);
                        acc += 10;
                    }
                    handle->set_done_flag();
                }
            };

        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
