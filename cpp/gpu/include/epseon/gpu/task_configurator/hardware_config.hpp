
#pragma once

#include "epseon/gpu/predecl.hpp"

#include <cstdint>
#include <memory>

namespace epseon {
    namespace gpu {
        namespace cpp {

            template <typename FP>
            struct HardwareConfig
                : public std::enable_shared_from_this<HardwareConfig<FP>> {

              public:
                uint32_t potential_buffer_size = {};
                uint32_t group_size            = {};
                uint32_t allocation_block_size = {};

              public:
                HardwareConfig(
                    uint32_t potential_buffer_size_,
                    uint32_t group_size_,
                    uint32_t allocation_block_size_
                ) :
                    potential_buffer_size(potential_buffer_size_),
                    group_size(group_size_),
                    allocation_block_size(allocation_block_size_) {}

              public: /* Public destructor. */
                // Virtual destructor.
                virtual ~HardwareConfig() = default;

                virtual std::shared_ptr<HardwareConfig> shared_clone() const {
                    return std::make_shared<HardwareConfig>(*this);
                }

                virtual std::unique_ptr<HardwareConfig> unique_clone() const {
                    return std::make_unique<HardwareConfig>(*this);
                }
            };
        } // namespace cpp
    }     // namespace gpu
} // namespace epseon
