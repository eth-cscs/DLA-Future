#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#


#
# To find CUDA:
#   1. Use `enable_language(CUDA)`
#   2. Set `CMAKE_CUDA_COMPILER` if cuda in custom directory
#
# Imported Targets:
#   dlaf::cudart
#   dlaf::cublas
#
cmake_minimum_required(VERSION 3.12)

find_path(CUDA_CUDART_INCLUDE cuda_runtime.h
          PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_path(CUDA_CUBLAS_INCLUDE cublas_v2.h
          PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_library(CUDA_CUDART_LIB cudart
             PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_CUBLAS_LIB cublas
             PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
mark_as_advanced(CUDA_CUDART_INCLUDE)
mark_as_advanced(CUDA_CUBLAS_INCLUDE)
mark_as_advanced(CUDA_CUDART_LIB)
mark_as_advanced(CUDA_CUBLAS_LIB)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDALIBS DEFAULT_MSG CUDA_CUDART_INCLUDE
                                                       CUDA_CUBLAS_INCLUDE
                                                       CUDA_CUDART_LIB
                                                       CUDA_CUBLAS_LIB)

if (CUDALIBS_FOUND AND NOT TARGET dlaf::cudart)
    add_library(dlaf::cudart IMPORTED INTERFACE)
    set_target_properties(dlaf::cudart PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUDA_CUDART_INCLUDE}"
      INTERFACE_LINK_LIBRARIES "${CUDA_CUDART_LIB}")
endif()

if (CUDALIBS_FOUND AND NOT TARGET dlaf::cublas)
    add_library(dlaf::cublas IMPORTED INTERFACE)
    set_target_properties(dlaf::cublas PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUDA_CUBLAS_INCLUDE}"
      INTERFACE_LINK_LIBRARIES "${CUDA_CUBLAS_LIB}")
endif()
