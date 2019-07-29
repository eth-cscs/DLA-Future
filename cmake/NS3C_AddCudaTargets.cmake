#
# NS3C
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

macro(NS3C_addCudaTargets)
  # check if the CUDA language is enabled
  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Please enable CUDA language before using this module")
  endif()

  ### CUDA
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

  mark_as_advanced(CUDA_INCLUDE_DIR CUDA_LIBRARY)

  set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIR})
  set(CUDA_LIBRARIES ${CUDA_LIBRARY})

  add_library(cuda::cuda INTERFACE IMPORTED)
  target_include_directories(cuda::cuda INTERFACE ${CUDA_INCLUDE_DIR})
  target_link_libraries(cuda::cuda INTERFACE ${CUDA_LIBRARY})

  ### CUBLAS
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

  mark_as_advanced(CUDA_CUBLAS_INCLUDE_DIR CUDA_CUBLAS_LIBRARY)

  set(CUDA_CUBLAS_INCLUDE_DIRS ${CUDA_CUBLAS_INCLUDE_DIR})
  set(CUDA_CUBLAS_LIBRARIES ${CUDA_CUBLAS_LIBRARY})

  add_library(cuda::cublas INTERFACE IMPORTED)
  target_include_directories(cuda::cublas INTERFACE ${CUDA_CUBLAS_INCLUDE_DIR})
  target_link_libraries(cuda::cublas INTERFACE ${CUDA_CUBLAS_LIBRARY})
endmacro()
