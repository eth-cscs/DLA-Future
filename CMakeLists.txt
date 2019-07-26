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

project(NS3C)

# ---------------------------------------------------------------------------
# CMake configurations
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(NS3C)

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
set(NS3C_WITH_CUDA OFF CACHE BOOL "Enable CUDA support")
if (NS3C_WITH_CUDA)
  enable_language(CUDA)

  # CUDA
  find_path(CUDA_INCLUDE_DIR
    cuda.h
    PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  find_library(CUDA_LIBRARY
    cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  )

  if (NOT CUDA_INCLUDE_DIR OR NOT CUDA_LIBRARY)
    message(FATAL_ERROR "Impossible to find cuda::cuda library")
  endif()

  add_library(cuda::cuda INTERFACE IMPORTED)
  target_include_directories(cuda::cuda INTERFACE ${CUDA_INCLUDE_DIR})
  target_link_libraries(cuda::cuda INTERFACE ${CUDA_LIBRARY})

  # CUBLAS
  find_path(CUDA_CUBLAS_INCLUDE_DIR
    cublas_v2.h
    PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  find_library(CUDA_CUBLAS_LIBRARY
    cublas
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  )

  if (NOT CUDA_CUBLAS_INCLUDE_DIR OR NOT CUDA_CUBLAS_LIBRARY)
    message(FATAL_ERROR "Impossible to find cuda::cublas library")
  endif()

  add_library(cuda::cublas INTERFACE IMPORTED)
  target_include_directories(cuda::cublas INTERFACE ${CUDA_CUBLAS_INCLUDE_DIR})
  target_link_libraries(cuda::cublas INTERFACE ${CUDA_CUBLAS_LIBRARY})
endif()

# ---------------------------------------------------------------------------
# Libraries configuration
# ---------------------------------------------------------------------------
# ----- MPI
find_package(MPI REQUIRED)

# ----- LAPACK/SCALAPACK
set(NS3C_WITH_MKL OFF CACHE BOOL "Enable MKL as provider for LAPACK")
if (NS3C_WITH_MKL)
  set(MKL_THREADING "Sequential" CACHE STRING "MKL Threading fixed to Sequential" FORCE)
  find_package(MKL REQUIRED COMPONENTS LAPACK)
  set(LAPACK_FOUND ${MKL_LAPACK_FOUND})

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
