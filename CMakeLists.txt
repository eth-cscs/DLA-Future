#
# NS3C
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

cmake_minimum_required(VERSION 3.12.4)

# ---------------------------------------------------------------------------
# PROJECT
# ---------------------------------------------------------------------------
project(NS3C C CXX)

set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
endif()

# ---------------------------------------------------------------------------
# CMake configurations
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(NS3C_Utils)

# ---------------------------------------------------------------------------
# Libraries configuration
# ---------------------------------------------------------------------------
# ----- MPI
find_package(MPI REQUIRED)

# ----- LAPACK/SCALAPACK
set(NS3C_WITH_MKL OFF CACHE BOOL "Enable MKL as provider for LAPACK")
if (NS3C_WITH_MKL)
  set(MKL_THREADING "Sequential" CACHE STRING "MKL Threading fixed to Sequential" FORCE)
  find_package(MKL REQUIRED)

  add_library(lapack::lapack ALIAS MKL::lapack)
else()
  find_package(LAPACK REQUIRED)
endif()

# ----- HPX
find_package(HPX REQUIRED)

# ----- EXTERNAL
add_subdirectory(external)

# ---------------------------------------------------------------------------
# NS3C library
# ---------------------------------------------------------------------------
add_library(NS3C INTERFACE)
target_include_directories(NS3C INTERFACE include)

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
set(NS3C_WITH_TEST BOOL "ON" "Enables tests")
if (NS3C_WITH_TEST)
  include(CTest)
  enable_testing()
  
  add_subdirectory(test)
endif()
