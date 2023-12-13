#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

#include <concepts>
#include <cstdint>
#include <memory>

namespace epseon::gpu::cpp {

    namespace scaling {

        class Base {
          public:
            Base()                = default;
            Base(const Base&)     = default;
            Base(Base&&) noexcept = default;

            virtual ~Base() = default;

            Base& operator=(const Base&)     = default;
            Base& operator=(Base&&) noexcept = default;

            [[nodiscard]] virtual uint64_t
            getAllocationTotalSizeBytes(uint64_t totalSizeBytes, uint64_t batchSize) const = 0;

            [[nodiscard]] virtual uint64_t getAllocationBufferCount(uint64_t batchSize) const = 0;
        };

        class BufferArray : public Base {
          public:
            BufferArray()                       = default;
            BufferArray(const BufferArray&)     = default;
            BufferArray(BufferArray&&) noexcept = default;

            ~BufferArray() override = default;

            BufferArray& operator=(const BufferArray&)     = default;
            BufferArray& operator=(BufferArray&&) noexcept = default;

            [[nodiscard]] uint64_t
            getAllocationTotalSizeBytes(uint64_t totalSizeBytes,
                                        uint64_t /*batchSize*/) const override {
                return totalSizeBytes;
            }

            [[nodiscard]] uint64_t getAllocationBufferCount(uint64_t batchSize) const override {
                return batchSize;
            }
        };

        class LargeBuffer : public Base {
          public:
            [[nodiscard]] uint64_t getAllocationTotalSizeBytes(uint64_t totalSizeBytes,
                                                               uint64_t batchSize) const override {
                return totalSizeBytes * batchSize;
            }

            [[nodiscard]] uint64_t getAllocationBufferCount(uint64_t /*batchSize*/) const override {
                return 1;
            }
        };
    }; // namespace scaling

} // namespace epseon::gpu::cpp