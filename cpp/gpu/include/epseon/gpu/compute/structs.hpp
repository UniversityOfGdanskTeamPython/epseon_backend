#pragma once

#include "epseon/vulkan_headers.hpp"

#include "epseon/gpu/compute/predecl.hpp"

namespace epseon::gpu::cpp {
    struct DescriptorSetWrite {
        vk::WriteDescriptorSet                writeInfo  = {};
        std::vector<vk::DescriptorBufferInfo> bufferInfo = {};
    };
} // namespace epseon::gpu::cpp