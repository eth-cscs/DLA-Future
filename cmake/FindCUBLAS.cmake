# 
# To find CUDA:
#   1. Use `enable_language(CUDA)`
#   2. Set `CMAKE_CUDA_COMPILER` if cuda in custom directory
#
# Imported Targets: 
#   dlaf::cublas
#
cmake_minimum_required(VERSION 3.12)

find_path(CUDA_TOOLKIT_INCLUDE cuda.h
          PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_library(CUDA_CUDART_LIB cudart
             PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_CUBLAS_LIB cublas
             PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
mark_as_advanced(CUDA_TOOLKIT_INCLUDE)
mark_as_advanced(CUDA_CUDART_LIB)
mark_as_advanced(CUDA_CUBLAS_LIB)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUBLAS DEFAULT_MSG CUDA_TOOLKIT_INCLUDE
                                                     CUDA_CUBLAS_LIB
                                                     CUDA_CUDART_LIB)

if (CUBLAS_FOUND AND NOT TARGET dlaf::cublas)
    add_library(dlaf::cublas IMPORTED INTERFACE)
    set_target_properties(dlaf::cublas PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_INCLUDE}"
      INTERFACE_LINK_LIBRARIES "${CUDA_CUDART_LIB};${CUDA_CUBLAS_LIB}")
endif()

