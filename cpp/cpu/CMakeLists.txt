cmake_minimum_required(VERSION 3.20)
project(
    epseon_cpu_project
    VERSION "0.1.0"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE epseon_cpu_SOURCE "${PROJECT_SOURCE_DIR}/source/*.c*")

add_library(
    epseon_cpu SHARED
    "${epseon_cpu_SOURCE}"
)
set(epseon_cpu_INCLUDE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/../include"
)
target_include_directories(epseon_cpu PUBLIC ${epseon_cpu_INCLUDE})
add_custom_command(
    TARGET epseon_cpu POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:epseon_cpu>"
        "${REPOSITORY_ROOT}/python/epseon_backend/device/cpu/_$<TARGET_FILE_NAME:epseon_cpu>"
)

file(GLOB_RECURSE TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

# Define the test executable
add_executable(epseon_cpu_tests ${TEST_SOURCES})
target_link_libraries(epseon_cpu_tests gtest_main epseon_cpu)
target_include_directories(epseon_cpu_tests PRIVATE ${epseon_cpu_INCLUDE})

# Add the tests to the GoogleTest framework
gtest_discover_tests(epseon_cpu_tests)
