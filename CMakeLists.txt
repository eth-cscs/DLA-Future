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
set(LAPACK_TARGET "")
set(DLAF_WITH_MKL OFF CACHE BOOL "Enable MKL as provider for LAPACK")
if (DLAF_WITH_MKL)
  set(MKL_CUSTOM_THREADING "Sequential")
  find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_SEQ)
  set(LAPACK_FOUND ${MKL_LAPACK_FOUND})

  set(LAPACK_TARGET "mkl::blas_32bit_seq")
else()
  set(LAPACK_TARGET "lapack::lapack")
  find_package(LAPACK REQUIRED)
endif()

# ----- HPX
find_package(HPX 1.4.0 REQUIRED)

# In HPX 1.3.0 there is compiler matching check, but it does not check matching of build types.
# Since it is source of headaches, waiting for HPX to add this check, we check on our own
get_property(_MULTI_CONFIG_GENERATOR GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (NOT _MULTI_CONFIG_GENERATOR)
  # They may not match just if none of them is a Debug build
  if (  (NOT HPX_BUILD_TYPE STREQUAL CMAKE_BUILD_TYPE)
          AND
        (HPX_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Debug")  )
    message(WARNING
      "The build type used (${CMAKE_BUILD_TYPE}) does not match with HPX build type (${HPX_BUILD_TYPE}).\n"
      "Please either change CMAKE_BUILD_TYPE to ${HPX_BUILD_TYPE} or use a ${CMAKE_BUILD_TYPE} build of HPX")
  endif()
else()
  # build-type check not available for multi-configuration generators
endif()

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
