#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was DLAFConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET DLAF)
  include(${CMAKE_CURRENT_LIST_DIR}/DLAF-Targets.cmake)
endif()

# enable custom modules to be used
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

# ===== VARIABLES
set(DLAF_WITH_OPENMP ON)
set(DLAF_WITH_CUDA ON)
set(DLAF_WITH_HIP OFF)
set(DLAF_WITH_GPU ON)
set(DLAF_WITH_SCALAPACK ON)
set(DLAF_WITH_HDF5 ON)

# ===== DEPENDENCIES
include(CMakeFindDependencyMacro)

# ---- CUDA/HIP
if(DLAF_WITH_CUDA)
  find_dependency(CUDAToolkit)
endif()

if(DLAF_WITH_HIP)
  find_dependency(rocblas)
  find_dependency(rocsolver)
endif()

if(DLAF_WITH_GPU)
  find_dependency(whip)
endif()

# ----- MPI
find_dependency(MPI)

# ----- OpenMP
if(DLAF_WITH_OPENMP)
  find_dependency(OpenMP)
endif()

# ----- LAPACK
set(DLAF_LAPACK_LIBRARY "-L/user-environment/linux-sles15-neoverse_v2/gcc-12.3.0/openblas-0.3.26-ubzfrukksiuzkkbiqawygvqmh6bt7lpz/lib -lopenblas -L/user-environment/linux-sles15-neoverse_v2/gcc-12.3.0/openblas-0.3.26-ubzfrukksiuzkkbiqawygvqmh6bt7lpz/lib -lopenblas")
set(DLAF_LAPACK_INCLUDE_DIR "")
find_dependency(DLAF_LAPACK)

# ----- ScaLAPACK
if(DLAF_WITH_SCALAPACK)
  set(DLAF_SCALAPACK_LIBRARY "-L/user-environment/linux-sles15-neoverse_v2/gcc-12.3.0/netlib-scalapack-2.2.0-mhjn6mjvjm6zt632zit4q3pxltpt4ajj/lib -lscalapack")
  set(DLAF_SCALAPACK_INCLUDE_DIR "")
  find_dependency(DLAF_SCALAPACK)
endif()

# ----- pika
find_dependency(pika PATHS /user-environment/linux-sles15-neoverse_v2/gcc-12.3.0/pika-0.29.0-ewknjlghclzzi6bvt2qa4rlq63dwxb6g/lib64/cmake/pika)

# ----- BLASPP/LAPACKPP
find_dependency(blaspp PATHS /user-environment/linux-sles15-neoverse_v2/gcc-12.3.0/blaspp-2024.05.31-n3bgocxpatl3zv7ba3acng2e34vh6lhw/lib64/cmake/blaspp)
find_dependency(lapackpp PATHS /user-environment/linux-sles15-neoverse_v2/gcc-12.3.0/lapackpp-2024.05.31-hmv5fjfhg7d2bfjgwgqabjnd7iunqmdj/lib64/cmake/lapackpp)

# ----- UMPIRE
find_dependency(Umpire PATHS /user-environment/linux-sles15-neoverse_v2/gcc-12.3.0/umpire-2024.02.0-nf66qidtyxjdtclcsb564q4inq3jz4da/lib64/cmake/umpire)

if(DLAF_WITH_HDF5)
  find_dependency(HDF5 PATHS /users/rmeli/spack/opt/spack/linux-sles15-neoverse_v2/gcc-12.3.0/hdf5-1.14.5-tnpjzl6yroi7xtfqpkolr6ymiwuo5fyq/cmake)
endif()

check_required_components(DLAF)
