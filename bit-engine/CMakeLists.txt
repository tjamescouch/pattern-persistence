cmake_minimum_required(VERSION 3.20)
# Renaming project from physics simulator to compute engine
project(ftt_bitwise_engine LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ---- Find system frameworks ----
find_library(METAL_LIB Metal)
find_library(FOUNDATION_LIB Foundation)

# ---- Kernel compilation (offline metallib, reproducible) ----
# Note: Renamed 'kernels/universe.metal' must be used here
set(KERNEL_SRC ${CMAKE_SOURCE_DIR}/kernels/universe.metal)
set(AIR_OUT   ${CMAKE_BINARY_DIR}/universe.air)
set(METALLIB  ${CMAKE_BINARY_DIR}/universe.metallib)

# Command to compile Metal source to intermediate AIR
add_custom_command(
  OUTPUT ${AIR_OUT}
  COMMAND xcrun -sdk macosx metal -c ${KERNEL_SRC} -o ${AIR_OUT}
  DEPENDS ${KERNEL_SRC}
  VERBATIM)

# Command to link AIR to the final Metallib
add_custom_command(
  OUTPUT ${METALLIB}
  COMMAND xcrun -sdk macosx metallib ${AIR_OUT} -o ${METALLIB}
  DEPENDS ${AIR_OUT}
  VERBATIM)

add_custom_target(kernels ALL DEPENDS ${METALLIB})

# ---- Shared Library (The Bit-Engine Backend) ----
# CRITICAL: Replaced add_executable(sim ...) with add_library(... SHARED)
add_library(ftt_backend SHARED
  src/main.cpp
  src/gpu/metal/pipeline.mm
  src/gpu/metal/pipeline.h
)

# Ensure the library depends on the kernels being compiled first
add_dependencies(ftt_backend kernels)

target_include_directories(ftt_backend PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Provide the metallib path at compile time.
target_compile_definitions(ftt_backend PRIVATE METALLIB_PATH="${METALLIB}")

# Link against the necessary frameworks
target_link_libraries(ftt_backend PRIVATE ${METAL_LIB} ${FOUNDATION_LIB})

# Stricter warnings for quality control
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  target_compile_options(ftt_backend PRIVATE -Wall -Wextra -Wpedantic)
endif()