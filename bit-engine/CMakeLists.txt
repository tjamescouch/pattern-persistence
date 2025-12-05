cmake_minimum_required(VERSION 3.20)
project(qft_wavelet_automaton LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find system frameworks ----
find_library(METAL_LIB Metal)
find_library(FOUNDATION_LIB Foundation)

# ---- Kernel compilation (offline metallib, reproducible) ----
set(KERNEL_SRC ${CMAKE_SOURCE_DIR}/kernels/universe.metal)
set(AIR_OUT   ${CMAKE_BINARY_DIR}/universe.air)
set(METALLIB  ${CMAKE_BINARY_DIR}/universe.metallib)

add_custom_command(
  OUTPUT ${AIR_OUT}
  COMMAND xcrun -sdk macosx metal -c ${KERNEL_SRC} -o ${AIR_OUT}
  DEPENDS ${KERNEL_SRC}
  VERBATIM)

add_custom_command(
  OUTPUT ${METALLIB}
  COMMAND xcrun -sdk macosx metallib ${AIR_OUT} -o ${METALLIB}
  DEPENDS ${AIR_OUT}
  VERBATIM)

add_custom_target(kernels ALL DEPENDS ${METALLIB})

# ---- App ----
add_executable(sim
  src/main.cpp
  src/gpu/metal/pipeline.mm
  src/gpu/metal/pipeline.h
)

add_dependencies(sim kernels)
target_include_directories(sim PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Provide the metallib path at compile time.
target_compile_definitions(sim PRIVATE METALLIB_PATH="${METALLIB}")

target_link_libraries(sim PRIVATE ${METAL_LIB} ${FOUNDATION_LIB})

# Slightly stricter warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  target_compile_options(sim PRIVATE -Wall -Wextra -Wpedantic)
endif()

