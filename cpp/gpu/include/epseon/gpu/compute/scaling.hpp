#pragma once

#include "epseon/vulkan_headers.hpp"

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

            [[nodiscard]] virtual std::shared_ptr<scaling::Base> sharedClone() const = 0;
        };

        class BufferArray : public Base {
          public:
            BufferArray()                       = default;
            BufferArray(const BufferArray&)     = default;
            BufferArray(BufferArray&&) noexcept = default;

            ~BufferArray() override = default;

            BufferArray& operator=(const BufferArray&)     = default;
            BufferArray& operator=(BufferArray&&) noexcept = default;

            [[nodiscard]] std::shared_ptr<Base> sharedClone() const override {
                return std::make_shared<BufferArray>(*this);
            }

            [[nodiscard]] virtual uint64_t getAllocationTotalSizeBytes(uint64_t totalSizeBytes,
                                                                       uint64_t /*batchSize*/) {
                return totalSizeBytes;
            }

            [[nodiscard]] virtual uint64_t getAllocationBufferCount(uint64_t batchSize) {
                return batchSize;
            }
        };

        class LargeBuffer : public Base {
          public:
            [[nodiscard]] std::shared_ptr<Base> sharedClone() const override {
                return std::make_shared<LargeBuffer>(*this);
            }

            [[nodiscard]] virtual uint64_t getAllocationTotalSizeBytes(uint64_t totalSizeBytes,
                                                                       uint64_t batchSize) {
                return totalSizeBytes * batchSize;
            }

            [[nodiscard]] virtual uint64_t getAllocationBufferCount(uint64_t /*batchSize*/) {
                return 1;
            }
        };
    }; // namespace scaling

} // namespace epseon::gpu::cpp