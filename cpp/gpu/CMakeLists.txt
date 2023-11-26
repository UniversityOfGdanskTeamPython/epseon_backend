cmake_minimum_required(VERSION 3.20)
project(
    epseon_gpu_project
    VERSION "0.1.0"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(Python3 REQUIRED COMPONENTS Development Interpreter)

find_package(Vulkan REQUIRED)
message("----------------------------------- Vulkan ------------------------------------")
message("Vulkan_FOUND                           = '${Vulkan_FOUND}'")
message("Vulkan_VERSION                         = '${Vulkan_VERSION}'")
message("Vulkan_INCLUDE_DIRS                    = '${Vulkan_INCLUDE_DIRS}'")
message("Vulkan_INCLUDE_DIR                     = '${Vulkan_INCLUDE_DIR}'")
message("Vulkan_LIBRARIES                       = '${Vulkan_LIBRARIES}'")
message("Vulkan_LIBRARY                         = '${Vulkan_LIBRARY}'")
message("Vulkan_glslc_FOUND                     = '${Vulkan_glslc_FOUND}'")
message("Vulkan_glslangValidator_FOUND          = '${Vulkan_glslangValidator_FOUND}'")
message("Vulkan_glslang_FOUND                   = '${Vulkan_glslang_FOUND}'")
message("Vulkan_shaderc_combined_FOUND          = '${Vulkan_shaderc_combined_FOUND}'")
message("Vulkan_SPIRV-Tools_FOUND               = '${Vulkan_SPIRV-Tools_FOUND}'")
message("Vulkan_MoltenVK_FOUND                  = '${Vulkan_MoltenVK_FOUND}'")
message("Vulkan_volk_FOUND                      = '${Vulkan_volk_FOUND}'")
message("Vulkan_dxc_lib_FOUND                   = '${Vulkan_dxc_lib_FOUND}'")
message("Vulkan_dxc_exe_FOUND                   = '${Vulkan_dxc_exe_FOUND}'")
message("Vulkan_GLSLC_EXECUTABLE                = '${Vulkan_GLSLC_EXECUTABLE}'")
message("Vulkan_GLSLANG_VALIDATOR_EXECUTABLE    = '${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}'")
message("Vulkan_glslang_LIBRARY                 = '${Vulkan_glslang_LIBRARY}'")
message("Vulkan_shaderc_combined_LIBRARY        = '${Vulkan_shaderc_combined_LIBRARY}'")
message("Vulkan_SPIRV-Tools_LIBRARY             = '${Vulkan_SPIRV-Tools_LIBRARY}'")
message("Vulkan_MoltenVK_LIBRARY                = '${Vulkan_MoltenVK_LIBRARY}'")
message("Vulkan_volk_LIBRARY                    = '${Vulkan_volk_LIBRARY}'")
message("Vulkan_dxc_LIBRARY                     = '${Vulkan_dxc_LIBRARY}'")
message("Vulkan_dxc_EXECUTABLE                  = '${Vulkan_dxc_EXECUTABLE}'")
message("------------------------------------- end -------------------------------------")

file(GLOB_RECURSE epseon_gpu_SOURCE "${PROJECT_SOURCE_DIR}/source/*.c*")

add_library(
    epseon_gpu SHARED
    "${epseon_gpu_SOURCE}"
)
set(epseon_gpu_INCLUDE
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PRIVATE "${PROJECT_SOURCE_DIR}/../include"
    PRIVATE "${REPOSITORY_ROOT}/external/spdlog/include"
    PRIVATE "${REPOSITORY_ROOT}/external/fmt/include"
    PRIVATE "${REPOSITORY_ROOT}/external/vma/include"
    PRIVATE "${REPOSITORY_ROOT}/external/vma_hpp/include"
    PRIVATE "${REPOSITORY_ROOT}/external/pybind11/include"
    PRIVATE "${Python3_INCLUDE_DIRS}"
    PRIVATE "${Vulkan_INCLUDE_DIRS}"
)
set(epseon_gpu_LINK_LIBS
    PRIVATE dl
    PRIVATE fmt::fmt
    PRIVATE VulkanMemoryAllocator
    PRIVATE "${Python3_LIBRARIES}"
    PRIVATE "${Vulkan_LIBRARIES}"
)
target_include_directories(epseon_gpu
    ${epseon_gpu_INCLUDE}
)
target_link_libraries(epseon_gpu
    ${epseon_gpu_LINK_LIBS}
)


if(WIN32)
    add_custom_command(
        TARGET epseon_gpu POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:epseon_gpu>"
            "${REPOSITORY_ROOT}/python/epseon_backend/device/gpu/_libepseon_gpu.pyd"
    )
else()
    add_custom_command(
        TARGET epseon_gpu POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:epseon_gpu>"
            "${REPOSITORY_ROOT}/python/epseon_backend/device/gpu/_libepseon_gpu.so"
    )
endif()


file(GLOB_RECURSE files "${PROJECT_SOURCE_DIR}/test/*.cpp")
foreach(file ${files})
    get_filename_component(TEST_NAME ${file} NAME_WLE)
    add_executable(
        ${TEST_NAME}
        ${file}
    )
    target_link_libraries( ${TEST_NAME}
        PRIVATE gtest_main
        PRIVATE epseon_gpu
        ${epseon_gpu_LINK_LIBS}
    )
    target_include_directories( ${TEST_NAME}
        PRIVATE ${epseon_gpu_INCLUDE}
    )
    gtest_discover_tests(${TEST_NAME})
endforeach()
