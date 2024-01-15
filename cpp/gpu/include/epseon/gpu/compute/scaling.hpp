#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"
#include "epseon/gpu/compute/spirv.hpp"

#include <concepts>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace epseon::gpu::cpp {

    namespace scaling {

        class Base {
          public:
            explicit Base(uint32_t batchSize_) :
                batchSize(batchSize_) {}

            Base(const Base&)     = default;
            Base(Base&&) noexcept = default;

            virtual ~Base() = default;

            Base& operator=(const Base&)     = default;
            Base& operator=(Base&&) noexcept = default;

            [[nodiscard]] uint64_t getBatchSize() const {
                return this->batchSize;
            }

            [[nodiscard]] virtual MacroMapT getImpliedMacroDefs() const = 0;

            [[nodiscard]] virtual uint64_t
                getAllocationTotalSizeBytes(uint64_t /*totalSizeBytes*/) const = 0;

            [[nodiscard]] virtual uint64_t getAllocationBufferCount() const = 0;

          private:
            uint32_t batchSize;
        };

        class BufferArray : public Base {
          public:
            explicit BufferArray(uint32_t batchSize_) :
                Base(batchSize_) {}

            BufferArray(const BufferArray&)     = default;
            BufferArray(BufferArray&&) noexcept = default;

            ~BufferArray() override = default;

            BufferArray& operator=(const BufferArray&)     = default;
            BufferArray& operator=(BufferArray&&) noexcept = default;

            [[nodiscard]] MacroMapT getImpliedMacroDefs() const override {
                return {{"SCALING_LARGE_BUFFER", "0"},
                        {"SCALING_BUFFER_ARRAY", "1"},
                        {"BATCH_SIZE", std::to_string(getBatchSize())}};
            }

            [[nodiscard]] uint64_t
            getAllocationTotalSizeBytes(uint64_t totalSizeBytes) const override {

                return totalSizeBytes;
            }

            [[nodiscard]] uint64_t getAllocationBufferCount() const override {
                return getBatchSize();
            }
        };

        class LargeBuffer : public Base {
          public:
            explicit LargeBuffer(uint32_t batchSize_) :
                Base(batchSize_) {}

            LargeBuffer(const LargeBuffer&)     = default;
            LargeBuffer(LargeBuffer&&) noexcept = default;

            ~LargeBuffer() override = default;

            LargeBuffer& operator=(const LargeBuffer&)     = default;
            LargeBuffer& operator=(LargeBuffer&&) noexcept = default;

            [[nodiscard]] MacroMapT getImpliedMacroDefs() const override {
                return {{"SCALING_LARGE_BUFFER", "1"},
                        {"SCALING_BUFFER_ARRAY", "0"},
                        {"BATCH_SIZE", std::to_string(getBatchSize())}};
            }

            [[nodiscard]] uint64_t
            getAllocationTotalSizeBytes(uint64_t totalSizeBytes) const override {
                return totalSizeBytes * getBatchSize();
            }

            [[nodiscard]] uint64_t getAllocationBufferCount() const override {
                return 1;
            }
        };
    }; // namespace scaling

} // namespace epseon::gpu::cpp
