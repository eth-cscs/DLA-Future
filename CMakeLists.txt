#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

cmake_minimum_required(VERSION 3.14)

project(DLAF VERSION 0.1.0)

# ---------------------------------------------------------------------------
# CMake configurations
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(DLAF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build Type" FORCE)
endif()

option(BUILD_SHARED_LIBS "Build shared libraries." OFF)

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
# ----- C
enable_language(C)

# ----- CXX
enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)

# ---- CUDA
set(DLAF_WITH_CUDA OFF CACHE BOOL "Enable CUDA support")
if (DLAF_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDALIBS REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Libraries configuration
# ---------------------------------------------------------------------------
# ----- MPI
find_package(MPI REQUIRED)

# ----- LAPACK/SCALAPACK
set(LAPACK_TARGET "")
set(DLAF_WITH_MKL OFF CACHE BOOL "Enable MKL as provider for LAPACK")
if (DLAF_WITH_MKL)
  set(MKL_CUSTOM_THREADING "Sequential")
  find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_SEQ)

  set(LAPACK_TARGET "mkl::blas_32bit_seq")
else()
  set(LAPACK_TARGET "lapack::lapack")
  find_package(LAPACK REQUIRED)
endif()

# ----- HPX
find_package(HPX 1.4.0 REQUIRED)

# ----- BLASPP/LAPACKPP
find_package(OpenMP REQUIRED)
find_package(blaspp REQUIRED)
find_package(lapackpp REQUIRED)

# ----- EXTERNAL
add_subdirectory(external)

# ---------------------------------------------------------------------------
# DLAF library
# ---------------------------------------------------------------------------
add_subdirectory(src)

# ---------------------------------------------------------------------------
# mini Apps
# ---------------------------------------------------------------------------
set(DLAF_BUILD_MINIAPPS ON CACHE BOOL "Build miniapps")
if (DLAF_BUILD_MINIAPPS)
  add_subdirectory(miniapp)
endif()

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
set(DLAF_WITH_TEST ON CACHE BOOL "Enables tests")
if (DLAF_WITH_TEST)
  include(CTest)
  enable_testing()

  add_subdirectory(test)
endif()

# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------
option(BUILD_DOC "Build documentation" OFF)
if (BUILD_DOC)
  add_subdirectory(doc)
endif()
