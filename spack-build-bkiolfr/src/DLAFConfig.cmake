#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2023, ETH Zurich
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
set(DLAF_WITH_MKL OFF)
set(DLAF_WITH_CUDA OFF)
set(DLAF_WITH_HIP OFF)
set(DLAF_WITH_GPU )

# ===== DEPENDENCIES
include(CMakeFindDependencyMacro)

# ---- CUDA/HIP
if(DLAF_WITH_CUDA)
  find_dependency(CUDAToolkit)
endif()

if(DLAF_WITH_HIP)
  find_dependency(rocblas)
  find_dependency(rocprim)
  find_dependency(rocsolver)
  find_dependency(rocthrust)
endif()

# ----- MPI
find_dependency(MPI)

# ----- OpenMP
if(DLAF_WITH_OPENMP)
  find_dependency(OpenMP)
endif()

# ----- LAPACK
if(DLAF_WITH_MKL)
  set(MKL_ROOT "")
  set(MKL_CUSTOM_THREADING "")

  find_dependency(MKL)

  set(MKL_LAPACK_TARGET "")
  add_library(DLAF::LAPACK INTERFACE IMPORTED GLOBAL)
  target_link_libraries(DLAF::LAPACK INTERFACE ${MKL_LAPACK_TARGET})
else()
  set(LAPACK_LIBRARY "-L/home/rmeli/spack/opt/spack/linux-ubuntu23.04-zen4/gcc-12.2.0/openblas-0.3.23-2n3sn6c3q344pjxxw254dbayikqwj4el/lib -lopenblas -L/home/rmeli/spack/opt/spack/linux-ubuntu23.04-zen4/gcc-12.2.0/openblas-0.3.23-2n3sn6c3q344pjxxw254dbayikqwj4el/lib -lopenblas")

  find_dependency(LAPACK)
endif()

# ----- pika
find_dependency(pika PATHS /home/rmeli/spack/opt/spack/linux-ubuntu23.04-zen4/gcc-12.2.0/pika-0.15.0-67xcqvxrm6vdtiwlg7ydijxgmq46ieca/lib/cmake/pika)
find_dependency(pika-algorithms PATHS /home/rmeli/spack/opt/spack/linux-ubuntu23.04-zen4/gcc-12.2.0/pika-algorithms-0.1.2-ex2tk5xmmbenfvmq6ln7j6mkcaweunoh/lib/cmake/pika-algorithms)

# ----- BLASPP/LAPACKPP
find_dependency(blaspp PATHS /home/rmeli/spack/opt/spack/linux-ubuntu23.04-zen4/gcc-12.2.0/blaspp-2022.07.00-prnh2fcrtvkblkcet6gbgtoductgfm2g/lib/blaspp)
find_dependency(lapackpp PATHS /home/rmeli/spack/opt/spack/linux-ubuntu23.04-zen4/gcc-12.2.0/lapackpp-2022.07.00-gjn32b6nbdzxceosom7fgz242iimvqbg/lib/lapackpp)

# ----- UMPIRE
find_dependency(Umpire PATHS /home/rmeli/spack/opt/spack/linux-ubuntu23.04-zen4/gcc-12.2.0/umpire-2022.03.1-hfqps4kzylnywiemdy4se5tqxvrngcrc/lib/cmake/umpire)

if(DLAF_WITH_GPU)
  find_dependency(whip)
endif()

check_required_components(DLAF)
