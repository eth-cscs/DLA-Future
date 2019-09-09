#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

cmake_minimum_required(VERSION 3.12.4)

project(DLAF)

# ---------------------------------------------------------------------------
# CMake configurations
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(DLAF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
endif()

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

  DLAF_addCudaTargets()
endif()

# ---------------------------------------------------------------------------
# Libraries configuration
# ---------------------------------------------------------------------------
# ----- MPI
find_package(MPI REQUIRED)

# ----- LAPACK/SCALAPACK
set(DLAF_WITH_MKL OFF CACHE BOOL "Enable MKL as provider for LAPACK")
if (DLAF_WITH_MKL)
  set(MKL_CUSTOM_THREADING "Sequential")
  find_package(MKL REQUIRED COMPONENTS LAPACK)
  set(LAPACK_FOUND ${MKL_LAPACK_FOUND})

  add_library(lapack::lapack ALIAS MKL::lapack)
else()
  find_package(LAPACK REQUIRED)
endif()

# ----- HPX
find_package(HPX REQUIRED)

# ----- BLASPP/LAPACKPP
find_package(blaspp REQUIRED)
find_package(lapackpp REQUIRED)

# ----- EXTERNAL
add_subdirectory(external)

# ---------------------------------------------------------------------------
# DLAF library
# ---------------------------------------------------------------------------
add_subdirectory(dlaf)

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
