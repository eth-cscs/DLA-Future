#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

#
# To find CUDA, do one of the following:
#   - Set `CUDALIBS_ROOT` as an environment variable or as a CMake variable
#   - Set the environment variable `CUDA_HOME`
#   - Use `enable_language(CUDA)` and set `CMAKE_CUDA_COMPILER` if cuda in
#     a custom directory
#
# Imported Targets:
#   dlaf::cudart
#   dlaf::cublas
#   dlaf::cusolver
#
cmake_minimum_required(VERSION 3.12)

find_path(
  CUDA_CUDART_INCLUDE cuda_runtime.h
  PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $ENV{CUDA_HOME}
  PATH_SUFFIXES include
)
find_path(
  CUDA_CUBLAS_INCLUDE cublas_v2.h
  PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $ENV{CUDA_HOME}
  PATH_SUFFIXES include
)
find_path(
  CUDA_CUSOLVER_INCLUDE cusolverDn.h
  PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $ENV{CUDA_HOME}
  PATH_SUFFIXES include
)
find_library(
  CUDA_CUDART_LIB cudart
  PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} $ENV{CUDA_HOME}
  PATH_SUFFIXES lib64
)
find_library(
  CUDA_CUBLAS_LIB cublas
  PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} $ENV{CUDA_HOME}
  PATH_SUFFIXES lib64
)
find_library(
  CUDA_CUSOLVER_LIB cusolver
  PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} $ENV{CUDA_HOME}
  PATH_SUFFIXES lib64
)
mark_as_advanced(CUDA_CUDART_INCLUDE)
mark_as_advanced(CUDA_CUBLAS_INCLUDE)
mark_as_advanced(CUDA_CUSOLVER_INCLUDE)
mark_as_advanced(CUDA_CUDART_LIB)
mark_as_advanced(CUDA_CUBLAS_LIB)
mark_as_advanced(CUDA_CUSOLVER_LIB)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  CUDALIBS
  DEFAULT_MSG
  CUDA_CUDART_INCLUDE
  CUDA_CUBLAS_INCLUDE
  CUDA_CUSOLVER_INCLUDE
  CUDA_CUDART_LIB
  CUDA_CUBLAS_LIB
  CUDA_CUSOLVER_LIB
)

if(CUDALIBS_FOUND AND NOT TARGET dlaf::cudart)
  add_library(dlaf::cudart IMPORTED INTERFACE)
  set_target_properties(
    dlaf::cudart PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUDA_CUDART_INCLUDE}"
                            INTERFACE_LINK_LIBRARIES "${CUDA_CUDART_LIB}"
  )
endif()

if(CUDALIBS_FOUND AND NOT TARGET dlaf::cublas)
  add_library(dlaf::cublas IMPORTED INTERFACE)
  set_target_properties(
    dlaf::cublas PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUDA_CUBLAS_INCLUDE}"
                            INTERFACE_LINK_LIBRARIES "${CUDA_CUBLAS_LIB}"
  )
endif()

if(CUDALIBS_FOUND AND NOT TARGET dlaf::cusolver)
  add_library(dlaf::cusolver IMPORTED INTERFACE)
  set_target_properties(
    dlaf::cusolver PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUDA_CUSOLVER_INCLUDE}"
                              INTERFACE_LINK_LIBRARIES "${CUDA_CUSOLVER_LIB}"
  )
endif()
