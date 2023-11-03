cmake_minimum_required(VERSION 3.20)
project(
    epseon_gpu_project
    VERSION "0.1.0"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE epseon_gpu_SOURCE "${PROJECT_SOURCE_DIR}/source/*.c*")

add_library(
    epseon_gpu SHARED
    "${epseon_gpu_SOURCE}"
)
set(epseon_gpu_INCLUDE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/../include"
)
target_include_directories(epseon_gpu PUBLIC ${epseon_gpu_INCLUDE})
add_custom_command(
    TARGET epseon_gpu POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:epseon_gpu>"
        "${REPOSITORY_ROOT}/python/epseon_backend/device/gpu/_$<TARGET_FILE_NAME:epseon_gpu>"
)


file(GLOB_RECURSE TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

# Define the test executable
add_executable(epseon_gpu_tests ${TEST_SOURCES})
target_link_libraries(epseon_gpu_tests gtest_main epseon_gpu)
target_include_directories(epseon_gpu_tests PRIVATE ${epseon_gpu_INCLUDE})

# Add the tests to the GoogleTest framework
gtest_discover_tests(epseon_gpu_tests)
