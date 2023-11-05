cmake_minimum_required(VERSION 3.20)
project(
    epseon_cpu_project
    VERSION "0.1.0"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(Python3 REQUIRED COMPONENTS Development)
file(GLOB_RECURSE epseon_cpu_SOURCE "${PROJECT_SOURCE_DIR}/source/*.c*")

add_library(
    epseon_cpu SHARED
    "${epseon_cpu_SOURCE}"
)
set(epseon_cpu_INCLUDE
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PRIVATE "${PROJECT_SOURCE_DIR}/../include"
    PRIVATE "${Python3_INCLUDE_DIRS}"
)
set(epseon_cpu_LINK_LIBS
    "${Python3_LIBRARIES}"
)
target_include_directories(epseon_cpu
    ${epseon_cpu_INCLUDE}
)
target_link_libraries(epseon_cpu
    ${epseon_cpu_LINK_LIBS}
)

if(WIN32)
    add_custom_command(
        TARGET epseon_cpu POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:epseon_cpu>"
            "${REPOSITORY_ROOT}/python/epseon_backend/device/cpu/_libepseon_cpu.pyd"
    )
else()
    add_custom_command(
        TARGET epseon_cpu POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:epseon_cpu>"
            "${REPOSITORY_ROOT}/python/epseon_backend/device/cpu/_libepseon_cpu.so"
    )
endif()
file(GLOB_RECURSE TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

# Define the test executable
add_executable(epseon_cpu_tests ${TEST_SOURCES})
target_link_libraries(epseon_cpu_tests
    PRIVATE gtest_main
    PRIVATE epseon_cpu
    ${epseon_cpu_LINK_LIBS}
)
target_include_directories(epseon_cpu_tests
    PRIVATE ${epseon_cpu_INCLUDE}
)

# Add the tests to the GoogleTest framework
gtest_discover_tests(epseon_cpu_tests)
