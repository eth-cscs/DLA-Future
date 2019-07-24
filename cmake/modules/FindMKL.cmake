#
# NS3C
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# Find MKL library
#
# This module sets the following variables:
#  MKL_FOUND, LAPACK_FOUND, SCALAPACK_FOUND 
#     set to true if a library implementing the MKL interface is found
#  MKL_HAVE_NUM_THREADS_UTIL if it is available in the library
#
# Following options are allowed:
#   MKL_ROOT -  where to look for the library.
#               If not set, it uses the environment variable MKLROOT
#   MKL_THREADING - threading mode available for LAPACK
#   MKL_MPI_TYPE - mpi supporto for SCALAPACK
#
# It creates targets MKL::lapack and MKL::scalapack

set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "Intel MKL Path")

if (NOT MKL_ROOT)
  set(MKL_ROOT $ENV{MKLROOT})
endif()


### LAPACK
# ----- Options
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MKL_THREADING_OPTIONS Sequential "Intel OpenMP")
  set(MKL_THREADING_DEFAULT "Intel OpenMP")
else()
  set(MKL_THREADING_OPTIONS Sequential "GNU OpenMP" "Intel OpenMP")
  set(MKL_THREADING_DEFAULT "GNU OpenMP")
endif()

set(MKL_THREADING "${MKL_THREADING_DEFAULT}" CACHE STRING "MKL Threading support")
set_property(CACHE MKL_THREADING PROPERTY STRINGS ${MKL_THREADING_OPTIONS})

# ----- Look for the library
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MKL_LIB_DIR "-L${MKL_ROOT}/lib -Wl,-rpath,${MKL_ROOT}/lib")
else()
  set(MKL_LIB_DIR "-L${MKL_ROOT}/lib/intel64")
endif()

# ----- set MKL Threading
if(MKL_THREADING MATCHES "Sequential")
  set(MKL_THREAD_LIB "-lmkl_sequential")
elseif(MKL_THREADING MATCHES "GNU OpenMP")
  set(MKL_THREAD_LIB "-lmkl_gnu_thread -fopenmp")
elseif(MKL_THREADING MATCHES "Intel OpenMP")
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(INTEL_LIBS_ROOT "/opt/intel/lib" CACHE PATH "Path to Intel libraries")
    find_library(IOMP5_LIB iomp5 HINTS "${INTEL_LIBS_ROOT}" NO_DEFAULT_PATH)
    if (NOT IOMP5_LIB)
      message(FATAL_ERROR "libiomp5 not found, please set INTEL_LIBS_ROOT correctly")
    endif()
    set(IOMP5_LIB_INTERNAL "-Wl,-rpath,${INTEL_LIBS_ROOT} ${IOMP5_LIB}")
  else()
    set(IOMP5_LIB_INTERNAL "-liomp5")
  endif()
  set(MKL_THREAD_LIB "-lmkl_intel_thread ${IOMP5_LIB_INTERNAL}")
endif()
message(STATUS "MKL Threading: ${MKL_THREADING}")

set(LAPACK_INCLUDE_DIR "${MKL_ROOT}/include" CACHE PATH "LAPACK includes" FORCE)
# TODO pthread, m, dl ???
set(LAPACK_LIBRARY
  "${MKL_LIB_DIR} -lmkl_intel_lp64 ${MKL_THREAD_LIB} -lmkl_core -lpthread -lm -ldl"
  CACHE STRING "LAPACK libraries" FORCE)

mark_as_advanced(
  LAPACK_INCLUDE_DIR
  LAPACK_LIBRARY
)


### SCALAPACK
# ----- Options
if (UNIX)
  set(MKL_MPI_TYPE "IntelMPI" CACHE STRING "MKL MPI support")
  set_property(CACHE MKL_MPI_TYPE PROPERTY STRINGS "IntelMPI" "OpenMPI")
endif()

# ----- set MPI support
if (UNIX)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MKL_BLACS_LIB "-lmkl_blacs_mpich_lp64")
  else()
    if(MKL_MPI_TYPE MATCHES "OpenMPI")
      set(MKL_BLACS_LIB "-lmkl_blacs_openmpi_lp64")
    elseif(MKL_MPI_TYPE MATCHES "IntelMPI")
      set(MKL_BLACS_LIB "-lmkl_blacs_intelmpi_lp64")
    else()
      message(FATAL_ERROR "Unknown MKL MPI Support: ${MKL_MPI_TYPE}")
    endif()
  endif()
endif()

set(SCALAPACK_LIBRARY "-lmkl_scalapack_lp64 ${MKL_BLACS_LIB}" CACHE STRING "Scalapack libraries" FORCE)

mark_as_advanced(SCALAPACK_LIBRARY)


### Additional Option
include(CheckCXXSymbolExists)
CHECK_CXX_SYMBOL_EXISTS(mkl_get_max_threads mkl.h _MKL_HAVE_NUM_THREADS_UTIL)
if (_MKL_HAVE_NUM_THREADS_UTIL)
  set(MKL_HAVE_NUM_THREADS_UTIL TRUE)
endif()


### Checks
include(CMakePushCheckState)
cmake_push_check_state(RESET)

include(CheckFunctionExists)

# ----- LAPACK
cmake_reset_check_state()
set(CMAKE_REQUIRED_INCLUDES ${LAPACK_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARY})

unset(LAPACK_CHECK_BLAS CACHE)
check_symbol_exists(dgemm "mkl_blas.h" LAPACK_CHECK_BLAS)
if (NOT LAPACK_CHECK_BLAS)
  message(FATAL_ERROR "BLAS symbol not found with this configuration")
endif()

unset(LAPACK_CHECK CACHE)
check_symbol_exists(dpotrf_ "mkl_lapack.h" LAPACK_CHECK)
if (NOT LAPACK_CHECK)
  message(FATAL_ERROR "LAPACK symbol not found with this configuration")
endif()

# ----- SCALAPACK
cmake_reset_check_state()
set(CMAKE_REQUIRED_LIBRARIES ${SCALAPACK_LIBRARY})

unset(SCALAPACK_CHECK CACHE)
check_symbol_exists(pdpotrf_ "mkl_scalapack.h" SCALAPACK_CHECK)
if (NOT SCALAPACK_CHECK)
  message(FATAL_ERROR "Scalapack symbol not found with this configuration")
endif()

cmake_pop_check_state()


### MKL Package
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(LAPACK DEFAULT_MSG
  MKL_ROOT
  LAPACK_INCLUDE_DIR
  LAPACK_LIBRARY
)

find_package_handle_standard_args(SCALAPACK DEFAULT_MSG
  MKL_ROOT
  SCALAPACK_LIBRARY
)

find_package_handle_standard_args(MKL DEFAULT_MSG
  MKL_ROOT
  LAPACK_FOUND
  SCALAPACK_FOUND
)

# ----- LAPACK
if (LAPACK_FOUND)
  set(LAPACK_INCLUDE_DIRS ${LAPACK_INCLUDE_DIR})
  set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})

  if (NOT TARGET MKL::lapack)
    add_library(MKL::lapack INTERFACE IMPORTED GLOBAL)
  endif()

  target_include_directories(MKL::lapack
    INTERFACE
      ${LAPACK_INCLUDE_DIRS}
  )

  target_link_libraries(MKL::lapack
    INTERFACE
      ${LAPACK_LIBRARIES}
  )
endif()

# ----- SCALAPACK
if (SCALAPACK_FOUND)
  set(SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARY})

  if (NOT TARGET MKL::scalapack)
    add_library(MKL::scalapack INTERFACE IMPORTED GLOBAL)
  endif()

  target_link_libraries(MKL::scalapack
    INTERFACE
      ${SCALAPACK_LIBRARIES}
  )
endif()
