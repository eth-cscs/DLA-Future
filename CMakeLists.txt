#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

cmake_minimum_required(VERSION 3.16)

project(DLAF VERSION 0.1.0)

# ---------------------------------------------------------------------------
# CMake configurations
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(CTest)
include(DLAF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build Type" FORCE)
endif()

include(CMakeDependentOption)

option(BUILD_SHARED_LIBS "Build shared libraries." OFF)
option(DLAF_WITH_MKL "Enable MKL as provider for LAPACK" OFF)
option(DLAF_WITH_CUDA "Enable CUDA support" OFF)
cmake_dependent_option(DLAF_WITH_CUDA_MPI_RDMA "Enable MPI CUDA RDMA" OFF "DLAF_WITH_CUDA" OFF)
option(DLAF_WITH_COVERAGE "Enable coverage" OFF)
option(DLAF_BUILD_MINIAPPS "Build miniapps" ON)
option(DLAF_BUILD_TESTING "Build tests" ON)
option(DLAF_BUILD_DOC "Build documentation" OFF)
option(DLAF_WITH_PRECOMPILED_HEADERS "Use precompiled headers." OFF)

# Add color to ninja output
if(CMAKE_GENERATOR MATCHES "Ninja")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
  endif()
endif()

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
# ----- C
enable_language(C)

# ----- CXX
enable_language(CXX)

# ---- CUDA
if(DLAF_WITH_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;80" CACHE STRING "Cuda architectures" FORCE)
  endif()
  enable_language(CUDA)
  find_package(CUDALIBS REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
if(DLAF_WITH_COVERAGE)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -fprofile-arcs -ftest-coverage)
    add_link_options(-fprofile-arcs -ftest-coverage)
  else()
    message(FATAL_ERROR "Coverage can be enabled only for Debug builds")
  endif()
endif()

# ---------------------------------------------------------------------------
# Libraries configuration
# ---------------------------------------------------------------------------
# ----- MPI
find_package(MPI REQUIRED)

# ----- LAPACK/SCALAPACK
if(DLAF_WITH_MKL)
  find_package(MKL REQUIRED)

  set(MKL_LAPACK_TARGET "mkl::mkl_intel_32bit_seq_dyn"
      CACHE STRING "MKL LAPACK target (see FindMKL for details)"
  )

  # Note:
  # If specified targets are found, "standardize" them

  if(TARGET ${MKL_LAPACK_TARGET})
    set(LAPACK_FOUND TRUE)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED GLOBAL)
    target_link_libraries(LAPACK::LAPACK INTERFACE ${MKL_LAPACK_TARGET})
  else()
    message(FATAL_ERROR "${MKL_LAPACK_TARGET} as LAPACK target has not been found")
  endif()
else()
  find_package(LAPACK REQUIRED)
endif()

# ----- pika
find_package(pika REQUIRED)

# ----- BLASPP/LAPACKPP
find_package(blaspp REQUIRED)
find_package(lapackpp REQUIRED)

# ----- UMPIRE
find_package(Umpire REQUIRED)

# ----- EXTERNAL
add_subdirectory(external)

# ---------------------------------------------------------------------------
# DLAF library
# ---------------------------------------------------------------------------
add_subdirectory(src)

# ---------------------------------------------------------------------------
# mini Apps
# ---------------------------------------------------------------------------
if(DLAF_BUILD_MINIAPPS)
  add_subdirectory(miniapp)
endif()

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if(DLAF_BUILD_TESTING)
  add_subdirectory(test)
endif()

# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------
if(DLAF_BUILD_DOC)
  add_subdirectory(doc)
endif()
