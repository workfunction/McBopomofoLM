# Static arm64 build of ggml (CPU backend only) + upstream libslothe for McBopomofoLM.
# Recipe mirrors scratch/ime-lm-poc/sloth/build.sh (Metal off, BLAS off), but static.
# Driven by build_deps.sh; the Xcode target links the resulting .a files.
cmake_minimum_required(VERSION 3.20)
project(slothe_deps C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
add_subdirectory(ThirdParty/ggml ggml)

# upstream libslothe forward pass (no main()) with the McBopomofoLM local patches
# (ThirdParty/slothe/slothe-graph-cache.patch, slothe-load-errors.patch).
add_library(slothe STATIC ThirdParty/slothe/slothe.cpp)
target_compile_definitions(slothe PRIVATE SLOTHE_NO_MAIN)
target_include_directories(slothe PUBLIC ThirdParty/slothe)
target_link_libraries(slothe PUBLIC ggml ggml-base ggml-cpu)
