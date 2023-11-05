cmake_minimum_required(VERSION 3.20)
project(
    epseon_backend_core
    VERSION "0.1.0"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(Python3 REQUIRED COMPONENTS Development)
message("----------------------------------- Python3 -----------------------------------")
message("Python3_FOUND              = '${Python3_FOUND}'")
message("Python3_Interpreter_FOUND  = '${Python3_Interpreter_FOUND}'")
message("Python3_EXECUTABLE         = '${Python3_EXECUTABLE}'")
message("Python3_INTERPRETER_ID     = '${Python3_INTERPRETER_ID}'")
message("Python3_STDLIB             = '${Python3_STDLIB}'")
message("Python3_STDARCH            = '${Python3_STDARCH}'")
message("Python3_SITELIB            = '${Python3_SITELIB}'")
message("Python3_SITEARCH           = '${Python3_SITEARCH}'")
message("Python3_SOABI              = '${Python3_SOABI}'")
message("Python3_SOSABI             = '${Python3_SOSABI}'")
message("--------------------------------- development ---------------------------------")
message("Python3_Development_FOUND              = '${Python3_Development_FOUND}'")
message("                   .Module_FOUND       = '${Python3_Development.Module_FOUND}'")
message("                   .SABIModule_FOUND   = '${Python3_Development.SABIModule_FOUND}'")
message("                   .Embed_FOUND        = '${Python3_Development.Embed_FOUND}'")
message("Python3_INCLUDE_DIRS                   = '${Python3_INCLUDE_DIRS}'")
message("Python3_LINK_OPTIONS                   = '${Python3_LINK_OPTIONS}'")
message("Python3_LIBRARY                        = '${Python3_LIBRARY}'")
message("Python3_LIBRARY_DIRS                   = '${Python3_LIBRARY_DIRS}'")
message("Python3_LIBRARIES                      = '${Python3_LIBRARIES}'")
message("Python3_SABI_LIBRARIES                 = '${Python3_SABI_LIBRARIES}'")
message("Python3_RUNTIME_SABI_LIBRARY_DIRS      = '${Python3_RUNTIME_SABI_LIBRARY_DIRS}'")
message("------------------------------------- end -------------------------------------")

add_subdirectory("gpu")
add_subdirectory("cpu")
