#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

cmake_minimum_required(VERSION 3.22)

project(DLAF VERSION 0.1.0)

# ---------------------------------------------------------------------------
# CMake configurations
# ---------------------------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(CTest)
include(DLAF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build Type" FORCE)
endif()

include(CMakeDependentOption)

option(BUILD_SHARED_LIBS "Build shared libraries." ON)
option(DLAF_WITH_MKL "Enable MKL as provider for LAPACK" OFF)
option(DLAF_WITH_CUDA "Enable CUDA support" OFF)
option(DLAF_WITH_HIP "Enable HIP support" OFF)
cmake_dependent_option(DLAF_WITH_CUDA_MPI_RDMA "Enable MPI CUDA RDMA" OFF "DLAF_WITH_CUDA" OFF)
option(DLAF_WITH_COVERAGE "Enable coverage" OFF)
option(DLAF_BUILD_MINIAPPS "Build miniapps" ON)
option(DLAF_BUILD_TESTING "Build tests" ON)
option(DLAF_BUILD_DOC "Build documentation" OFF)
option(DLAF_WITH_PRECOMPILED_HEADERS "Use precompiled headers." OFF)

if(BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE "ON" CACHE BOOL "Enable position independent code" FORCE)
endif()

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

# ---- CUDA/HIP
if(DLAF_WITH_CUDA AND DLAF_WITH_HIP)
  message(FATAL_ERROR "DLAF_WITH_CUDA=ON and DLAF_WITH_HIP=ON. Only one of "
                      "them can be enabled at the same time."
  )
endif()
if(DLAF_WITH_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;80" CACHE STRING "Cuda architectures" FORCE)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(DLAF_WITH_GPU ON)
elseif(DLAF_WITH_HIP)
  set(CMAKE_HIP_EXTENSIONS OFF)
  enable_language(HIP)
  find_package(rocblas REQUIRED)
  find_package(rocprim REQUIRED)
  find_package(rocsolver REQUIRED)
  find_package(rocthrust REQUIRED)
  set(DLAF_WITH_GPU ON)
endif()

if(DLAF_WITH_GPU)
  find_package(whip REQUIRED)
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
    add_library(DLAF::LAPACK INTERFACE IMPORTED GLOBAL)
    target_link_libraries(DLAF::LAPACK INTERFACE ${MKL_LAPACK_TARGET})
  else()
    message(FATAL_ERROR "${MKL_LAPACK_TARGET} as LAPACK target has not been found")
  endif()
else()
  find_package(LAPACK REQUIRED)
endif()

# ----- pika
find_package(pika 0.15.0 REQUIRED)
find_package(pika-algorithms 0.1.1 REQUIRED)

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
  # Create a "do-nothing" DLAFConfig.cmake, and make it usable by find_package
  # by adding it in CMAKE_PREFIX_PATH.
  # In this way any call to `find_package(DLAF)` is a no-op. Indeed, nothing
  # is necessary to be done, since we are in-project and targets are still
  # available thanks to manually created aliases for exported targets.
  set(FAKE_INSTALL_DIR ${PROJECT_BINARY_DIR}/fake)
  file(MAKE_DIRECTORY ${FAKE_INSTALL_DIR})
  file(TOUCH ${FAKE_INSTALL_DIR}/DLAFConfig.cmake)
  list(PREPEND CMAKE_PREFIX_PATH ${FAKE_INSTALL_DIR})

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
