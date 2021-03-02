#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
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

include(CTest)
include(DLAF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build Type" FORCE)
endif()

option(BUILD_SHARED_LIBS "Build shared libraries." OFF)
option(DLAF_WITH_MKL "Enable MKL as provider for LAPACK" OFF)
option(DLAF_WITH_CUDA "Enable CUDA support" OFF)
option(DLAF_BUILD_MINIAPPS "Build miniapps" ON)
option(DLAF_BUILD_TESTING "Build tests" ON)
option(DLAF_BUILD_DOC "Build documentation" OFF)

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
# ----- C
enable_language(C)

# ----- CXX
enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)

# ---- CUDA
if (DLAF_WITH_CUDA)
  find_package(CUDALIBS REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Libraries configuration
# ---------------------------------------------------------------------------
# ----- MPI
find_package(MPI REQUIRED)

# ----- LAPACK/SCALAPACK
set(LAPACK_TARGET "")
if (DLAF_WITH_MKL)
  set(MKL_CUSTOM_THREADING "Sequential")
  find_package(MKL REQUIRED)
  set(LAPACK_TARGET "mkl::mkl_intel_32bit_seq_dyn")
  if (NOT TARGET ${LAPACK_TARGET})
    message(FATAL_ERROR "MKL target ${LAPACK_TARGET} has not been found")
  endif()
else()
  set(LAPACK_TARGET "lapack::lapack")
  find_package(LAPACK REQUIRED)
endif()

# ----- HPX
find_package(HPX 1.6.0 REQUIRED)

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
if (DLAF_BUILD_MINIAPPS)
  add_subdirectory(miniapp)
endif()

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if (DLAF_BUILD_TESTING)
  add_subdirectory(test)
endif()

# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------
if (DLAF_BUILD_DOC)
  add_subdirectory(doc)
endif()
