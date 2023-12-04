#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_VULKAN_VERSION 1001000
#include "vk_mem_alloc.hpp"

#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

#include "epseon/gpu/libgpu.hpp"
