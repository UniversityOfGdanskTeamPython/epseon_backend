
#pragma once

#include "epseon/gpu/predecl.hpp"

#include <cstdint>
#include <memory>

namespace epseon::gpu::cpp {

    template <typename FP>
    struct HardwareConfig : public std::enable_shared_from_this<HardwareConfig<FP>> {
        static_assert(std::is_floating_point<FP>::value, "FP must be an floating-point type.");

      public:
        uint32_t potential_buffer_size = {};
        uint32_t group_size            = {};
        uint32_t allocation_block_size = {};

      public:
        HardwareConfig(
            uint32_t potential_buffer_size_, uint32_t group_size_, uint32_t allocation_block_size_
        ) :
            potential_buffer_size(potential_buffer_size_),
            group_size(group_size_),
            allocation_block_size(allocation_block_size_) {}

        // Default constructor.
        HardwareConfig() = default;

        // Copy constructor.
        HardwareConfig(const HardwareConfig&) = default;

        // Copy assignment operator.
        HardwareConfig& operator=(const HardwareConfig&) = default;

        // Move constructor.
        HardwareConfig(HardwareConfig&&) noexcept = default;

        // Move assignment operator.
        HardwareConfig& operator=(HardwareConfig&&) noexcept = default;

      public: /* Public destructor. */
        // Virtual destructor.
        virtual ~HardwareConfig() = default;

      public: /* Public methods. */
        bool operator==(const HardwareConfig<FP>& other) const {
            return (
                this->potential_buffer_size == other.potential_buffer_size &&
                this->group_size == other.group_size &&
                this->allocation_block_size == other.allocation_block_size
            );
        }

        [[nodiscard]] virtual std::shared_ptr<HardwareConfig> shared_clone() const {
            return std::make_shared<HardwareConfig>(*this);
        }

        [[nodiscard]] virtual std::unique_ptr<HardwareConfig> unique_clone() const {
            return std::make_unique<HardwareConfig>(*this);
        }

      public: /* Public getters. */
        [[nodiscard]] uint32_t getPotentialBufferSize() const {
            return this->potential_buffer_size;
        }

        [[nodiscard]] uint32_t getGroupSize() const {
            return this->group_size;
        }

        [[nodiscard]] uint32_t getAllocationBlockSize() const {
            return this->allocation_block_size;
        }
    };
} // namespace epseon::gpu::cpp
