cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0048 NEW)

project(
    epseon_backend
    VERSION "0.1.0"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(REPOSITORY_ROOT "${PROJECT_SOURCE_DIR}")

add_subdirectory("external/spdlog")
# Add GoogleTest as a subdirectory
add_subdirectory("external/googletest")

# Enable testing
enable_testing()

# Include the GoogleTest's cmake functions
include(GoogleTest)

add_subdirectory("cpp")
