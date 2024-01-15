#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

#include "fmt/format.h"
#include "shaderc/env.h"
#include "shaderc/shaderc.h"
#include "shaderc/shaderc.hpp"
#include "shaderc/status.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace epseon::gpu::cpp {

    class CRC64 {
        static const uint64_t INITIAL_CRC            = 0xFFFFFFFFFFFFFFFF;
        static const uint64_t CRC_FIRST_BYTE_MASK    = 0x00000000000000FF;
        static const uint64_t CRC_LAST_BYTE_MASK     = 0x8000000000000000;
        static const uint64_t LEFT_SHIFT_LAST_BYTE   = 56;
        static const uint64_t LEFT_SHIFT_ONE_BIT     = 1;
        static const uint64_t LEFT_SHIFT_BYTE_OFFSET = 8;
        static const uint64_t CRC_ARRAY_SIZE         = 256;
        static const uint64_t HEX_UINT_MAX_WIDTH     = 16;

      public:
        static const uint64_t DEFAULT_POLYNOMIAL = 0x42F0E1EBA9EA3693;

        CRC64() :
            polynomial(DEFAULT_POLYNOMIAL) {
            initializeTable();
        }

        explicit CRC64(uint64_t poly) :
            polynomial(poly) {
            initializeTable();
        }

        std::string computeHex(const std::span<const uint8_t>& data) {
            uint64_t          value = compute(data);
            std::stringstream stream;
            // Set to output in hexadecimal
            stream << std::hex;
            // Set to fill with zeros
            stream << std::setfill('0');
            // Set the width to 16, as uint64_t can have up to 16 hex digits
            stream << std::setw(HEX_UINT_MAX_WIDTH);
            // Output the value
            stream << value;
            return stream.str();
        }

        uint64_t compute(const std::span<const uint8_t>& data) {
            uint64_t crc = INITIAL_CRC;
            for (uint64_t byte : data) {
                uint8_t pos = (byte ^ (crc >> LEFT_SHIFT_LAST_BYTE)) & CRC_FIRST_BYTE_MASK;
                // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
                crc         = crcTable[pos] ^ (crc << LEFT_SHIFT_BYTE_OFFSET);
                // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
            }
            return ~crc;
        }

      private:
        uint64_t                             polynomial = {};
        std::array<uint64_t, CRC_ARRAY_SIZE> crcTable   = {};

        void initializeTable() {
            for (int i = 0; i < crcTable.size(); ++i) {
                uint64_t crc = static_cast<uint64_t>(i) << LEFT_SHIFT_LAST_BYTE;
                for (int j = 0; j < LEFT_SHIFT_BYTE_OFFSET; ++j) {
                    crc = (crc << LEFT_SHIFT_ONE_BIT) ^
                          (static_cast<bool>(crc & CRC_LAST_BYTE_MASK) ? polynomial : 0);
                }
                // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
                crcTable[i] = crc;
                // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
            }
        }
    };

    class SPIRV {
      public:
        SPIRV()                       = delete;
        SPIRV(const SPIRV& other)     = delete;
        SPIRV(SPIRV&& other) noexcept = default;

        virtual ~SPIRV() = default;

        SPIRV& operator=(const SPIRV& other)     = delete;
        SPIRV& operator=(SPIRV&& other) noexcept = default;

        SPIRV static fromGlslFile(std::string      filePath,
                                  const MacroMapT& macroDefs,
                                  bool             optimize = false) {
            std::fstream file(filePath, std::ios::ate | std::ios::binary);

            if (!file.is_open()) {
                auto message = fmt::format("Failed to open file: {}", filePath);
                throw std::runtime_error(message);
            }

            std::ostringstream buf;
            buf << file.rdbuf();

            return fromGlslSource(buf.str(), macroDefs, optimize);
        }

        SPIRV static fromGlslSource(const std::string& sourceCode,
                                    const MacroMapT&   macroDefs,
                                    bool               optimize = true) {
            std::string shaderName = CRC64().computeHex(
                {reinterpret_cast // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                 <const uint8_t*>(sourceCode.data()),
                 sourceCode.size()});

            return SPIRV{compileShaderSource(shaderName, sourceCode, macroDefs, optimize)};
        }

        // Compiles a shader to a SPIR-V binary. Returns the binary as
        // a vector of 32-bit words.
        static std::vector<uint32_t> compileShaderSource(const std::string& sourceName,
                                                         const std::string& source,
                                                         const MacroMapT&   macroDefs,
                                                         bool               optimize = true) {
            shaderc::Compiler       compiler;
            shaderc::CompileOptions options;

            for (const auto& [key, value] : macroDefs) {
                options.AddMacroDefinition(key, value);
            }
            options.SetTargetEnvironment(shaderc_target_env_vulkan, vk::ApiVersion12);

            if (optimize) {
                options.SetOptimizationLevel(
                    shaderc_optimization_level::shaderc_optimization_level_performance);
            }
            shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
                source, shaderc_shader_kind::shaderc_compute_shader, sourceName.c_str(), options);

            if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
                std::string message =
                    fmt::format("Compilation status: {}\nNumber of errors: {}\nError message: {}",
                                static_cast<uint64_t>(module.GetCompilationStatus()),
                                module.GetNumErrors(),
                                module.GetErrorMessage());
                throw std::runtime_error(message);
                return {};
            }

            return {module.cbegin(), module.cend()};
        }

        [[nodiscard]] std::vector<uint32_t>& getCode() {
            return this->bytecode;
        }

        [[nodiscard]] const std::vector<uint32_t>& getCode() const {
            return this->bytecode;
        }

      private:
        explicit SPIRV(const std::vector<uint32_t>& bytecode_) :
            bytecode(bytecode_) {}

        explicit SPIRV(std::vector<uint32_t>&& bytecode_) :
            bytecode(std::move(bytecode_)) {}

        std::vector<uint32_t> bytecode;
    };

    class GLSL {
      public:
        GLSL() = delete;

        explicit GLSL(std::string_view sourceCode_) :
            sourceCode(sourceCode_) {}

        GLSL(std::string_view sourceCode_, MacroMapT&& macroDefs_) :
            sourceCode(sourceCode_),
            macroDefs(macroDefs_) {}

        GLSL(const GLSL& other)     = delete;
        GLSL(GLSL&& other) noexcept = default;

        virtual ~GLSL() = default;

        GLSL& operator=(const GLSL& other)     = delete;
        GLSL& operator=(GLSL&& other) noexcept = default;

        [[nodiscard]] SPIRV compile() const {
            return SPIRV::fromGlslSource(getSource(), getMacroDefs());
        }

        [[nodiscard]] std::string& getSource() {
            return this->sourceCode;
        }

        [[nodiscard]] const std::string& getSource() const {
            return this->sourceCode;
        }

        [[nodiscard]] MacroMapT& getMacroDefs() {
            return this->macroDefs;
        }

        [[nodiscard]] const MacroMapT& getMacroDefs() const {
            return this->macroDefs;
        }

        void updateMacroDefs(const MacroMapT& source) {
            for (const auto& [key, value] : source) {
                this->macroDefs[key] = value;
            }
        }

      private:
        std::string sourceCode = {};
        MacroMapT   macroDefs  = {};
    };
} // namespace epseon::gpu::cpp
