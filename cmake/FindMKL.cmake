#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# Find MKL library
#
# Components (UPPERCASE):
#   LAPACK
#   SCALAPACK (requires LAPACK, automatically added if not specified)
#
# Without specifying any components, it will look just for LAPACK.
#
# This module sets the following variables:
#   - MKL_FOUND set to true if all selected required components are found
#   - MKL_HAVE_NUM_THREADS_UTIL if it is available in the library
#
#   Components-related variables:
#   - MKL_<COMPONENT>_FOUND se to true if <COMPOMENT> is found
#   - MKL_<COMPONENT>_INCLUDE_DIRS and MKL_<COMPONENT>_LIBRARIES for each component
#
# Following options are allowed (for setting options from code see next section):
#   - MKL_ROOT - where to look for the library. If not set, it uses the environment variable MKLROOT
#
#   for each COMPONENT:
#   - MKL_<COMPONENT>_INCLUDE_DIR, MKL_<COMPONENT>_LIBRARY for each component
#
#   for LAPACK:
#   - MKL_THREADING - threading mode available for LAPACK
#
#   fo SCALAPACK:
#   - MKL_MPI_TYPE - MPI support for SCALAPACK
#
# If you want to set options from CMake script you cannot use previous variables, instead you should use:
#   - MKL_<COMPONENT>_CUSTOM_INCLUDE_DIR
#   - MKL_<COMPONENT>_CUSTOM_LIBRARY
#   - MKL_CUSTOM_THREADING
#   - MKL_CUSTOM_MPI_TYPE
#
# It creates targets MKL::lapack and MKL::scalapack (depending on selected components)

include(CMakePushCheckState)
include(CheckFunctionExists)
include(CheckCXXSymbolExists)
include(FindPackageHandleStandardArgs)

### helper function
function(check_valid_option selected_option)
  set(available_options ${ARGN})
  list(LENGTH available_options _how_many_options)

  if (NOT _how_many_options GREATER_EQUAL 1)
    message(
      FATAL_ERROR
      "You are checking value of an option without giving the list of valid ones")
  endif()

  list(FIND available_options ${selected_option} selected_index)
  if (${selected_index} EQUAL -1)
    message(FATAL_ERROR
      "You have selected '${selected_option}', but you have to choose among '${available_options}'")
  endif()
endfunction()

### helper functions: find components
macro(_mkl_find component_name)
  if (${component_name} STREQUAL "LAPACK")
    _mkl_find_lapack()
  elseif(${component_name} STREQUAL "SCALAPACK")
    _mkl_find_scalapack()
  else()
    message(FATAL_ERROR "Unknown component ${component_name}")
  endif()
endmacro()

macro(_mkl_find_lapack)
  # ----- Options
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MKL_THREADING_OPTIONS "Sequential" "Intel OpenMP")
    set(MKL_THREADING_DEFAULT "Intel OpenMP")
  else()
    set(MKL_THREADING_OPTIONS "Sequential" "GNU OpenMP" "Intel OpenMP")
    set(MKL_THREADING_DEFAULT "GNU OpenMP")
  endif()

  # allow to set option from script
  if (MKL_CUSTOM_THREADING)
    set(MKL_THREADING
      ${MKL_CUSTOM_THREADING}
      CACHE STRING
      "MKL Threading support (from code)"
      FORCE
    )
  else()
    set(MKL_THREADING ${MKL_THREADING_DEFAULT} CACHE STRING "MKL Threading support")
  endif()
  set_property(CACHE MKL_THREADING PROPERTY STRINGS ${MKL_THREADING_OPTIONS})
  check_valid_option(${MKL_THREADING} ${MKL_THREADING_OPTIONS})

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

  # allow to set LAPACK_INCLUDE_DIR from script
  if (MKL_LAPACK_CUSTOM_INCLUDE_DIR)
    unset(MKL_LAPACK_INCLUDE_DIR CACHE)
    find_path(MKL_LAPACK_INCLUDE_DIR
      mkl_lapack.h
      PATHS ${MKL_LAPACK_CUSTOM_INCLUDE_DIR}
      NO_DEFAULT_PATH)
  else()
    find_path(MKL_LAPACK_INCLUDE_DIR
      mkl_lapack.h
      PATHS ${MKL_ROOT}/include)
  endif()

  # allow to set LAPACK_LIBRARY from script
  if (MKL_LAPACK_CUSTOM_LIBRARY)
    set(MKL_LAPACK_LIBRARY
      ${MKL_LAPACK_CUSTOM_LIBRARY}
      CACHE STRING "LAPACK libraries (from code)"
      FORCE
    )
  else()
    set(MKL_LAPACK_LIBRARY
      "${MKL_LIB_DIR} -lmkl_intel_lp64\
      ${MKL_THREAD_LIB} -lmkl_core\
      -lpthread -lm -ldl"                            # TODO pthread, m, dl ???
      CACHE STRING "LAPACK libraries"
    )
  endif()

  mark_as_advanced(
    MKL_LAPACK_INCLUDE_DIR
    MKL_LAPACK_LIBRARY
  )

  find_package_handle_standard_args(MKL_LAPACK DEFAULT_MSG
    MKL_LAPACK_LIBRARY
    MKL_LAPACK_INCLUDE_DIR
  )
endmacro()

macro(_mkl_find_scalapack)
  # ----- Options
  if (UNIX)
    set(MKL_MPI_TYPE_OPTIONS "IntelMPI" "OpenMPI")
    set(MKL_MPI_TYPE_DEFAULT "IntelMPI")

    # allow to set MPI_MPI_TYPE from code
    if (MKL_CUSTOM_MPI_TYPE)
      set(MKL_MPI_TYPE ${MKL_CUSTOM_MPI_TYPE} CACHE STRING "MKL MPI support (from code)" FORCE)
    else()
      set(MKL_MPI_TYPE ${MKL_MPI_TYPE_DEFAULT} CACHE STRING "MKL MPI support")
    endif()
    set_property(CACHE MKL_MPI_TYPE PROPERTY STRINGS ${MKL_MPI_TYPE_OPTIONS})
    check_valid_option(${MKL_MPI_TYPE} ${MKL_MPI_TYPE_OPTIONS})
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

  # allow to set LAPACK_LIBRARY from script
  if (MKL_SCALAPACK_CUSTOM_LIBRARY)
    set(MKL_SCALAPACK_LIBRARY
      ${MKL_SCALAPACK_CUSTOM_LIBRARY}
      CACHE STRING
      "Scalapack libraries (from code)"
      FORCE
    )
  else()
    set(MKL_SCALAPACK_LIBRARY
      "-lmkl_scalapack_lp64 ${MKL_BLACS_LIB}"
      CACHE STRING "Scalapack libraries"
    )
  endif()

  mark_as_advanced(MKL_SCALAPACK_LIBRARY)

  find_package_handle_standard_args(MKL_SCALAPACK DEFAULT_MSG
    MKL_SCALAPACK_LIBRARY
  )
endmacro()

### helper functions: checks components
macro(_mkl_check component_name)
  if (${component_name} STREQUAL "LAPACK")
    _mkl_check_lapack()
  elseif(${component_name} STREQUAL "SCALAPACK")
    _mkl_check_scalapack()
  else()
    message(FATAL_ERROR "Unknown component ${component_name}")
  endif()
endmacro()

function(_mkl_check_lapack)
  cmake_push_check_state(RESET)

  set(CMAKE_REQUIRED_INCLUDES ${MKL_LAPACK_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${MKL_LAPACK_LIBRARY})

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

  cmake_pop_check_state()
endfunction()

function(_mkl_check_scalapack)
  cmake_push_check_state(RESET)

  set(CMAKE_REQUIRED_INCLUDES ${MKL_LAPACK_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${MKL_LAPACK_LIBRARY} ${MKL_SCALAPACK_LIBRARY})

  unset(SCALAPACK_CHECK CACHE)
  check_symbol_exists(pdpotrf_ "mkl_scalapack.h" SCALAPACK_CHECK)
  if (NOT SCALAPACK_CHECK)
    message(FATAL_ERROR "Scalapack symbol not found with this configuration")
  endif()

  cmake_pop_check_state()
endfunction()


### MAIN
set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "Intel MKL Path")

find_path(MKL_INCLUDE_DIR
  mkl.h
  PATHS ${MKL_ROOT}/include
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MKL_LIB_DIR "-L${MKL_ROOT}/lib -Wl,-rpath,${MKL_ROOT}/lib")
else()
  set(MKL_LIB_DIR "-L${MKL_ROOT}/lib/intel64")
endif()

### Additional Option
cmake_push_check_state(RESET)

set(CMAKE_REQUIRED_INCLUDES ${MKL_INCLUDE_DIR})

unset(_MKL_HAVE_NUM_THREADS_UTIL CACHE)
check_cxx_symbol_exists(mkl_get_max_threads mkl.h _MKL_HAVE_NUM_THREADS_UTIL)
if (_MKL_HAVE_NUM_THREADS_UTIL)
  set(MKL_HAVE_NUM_THREADS_UTIL TRUE)
endif()

cmake_pop_check_state()

### Components
if (NOT MKL_FIND_COMPONENTS)
  set(MKL_FIND_COMPONENTS LAPACK)
endif()

list(FIND MKL_FIND_COMPONENTS SCALAPACK _MKL_WITH_SCALAPACK)
if (_MKL_WITH_SCALAPACK)
  list(APPEND MKL_FIND_COMPONENTS LAPACK)
endif()

list(REMOVE_DUPLICATES MKL_FIND_COMPONENTS)
list(SORT MKL_FIND_COMPONENTS)  # LAPACK should be found before SCALAPACK (exploit alphabetical order)
foreach(_mkl_component_name IN LISTS MKL_FIND_COMPONENTS)
  _mkl_find(${_mkl_component_name})
  _mkl_check(${_mkl_component_name})

  if (MKL_${_mkl_component_name}_FOUND)
    continue()
  else()
    message(FATAL_ERROR "!!! ${_mkl_component_name} not found")
  endif()
endforeach()

### MKL Package
find_package_handle_standard_args(MKL
  FOUND_VAR MKL_FOUND
  REQUIRED_VARS
    MKL_ROOT
  HANDLE_COMPONENTS
)

# ----- LAPACK
if (MKL_LAPACK_FOUND)
  set(MKL_LAPACK_INCLUDE_DIRS ${MKL_LAPACK_INCLUDE_DIR})
  set(MKL_LAPACK_LIBRARIES ${MKL_LAPACK_LIBRARY})

  if (NOT TARGET MKL::lapack)
    add_library(MKL::lapack INTERFACE IMPORTED GLOBAL)
  endif()

  target_include_directories(MKL::lapack
    INTERFACE
      ${MKL_LAPACK_INCLUDE_DIRS}
  )

  target_link_libraries(MKL::lapack
    INTERFACE
      ${MKL_LAPACK_LIBRARIES}
  )
endif()

# ----- SCALAPACK
if (MKL_SCALAPACK_FOUND)
  set(MKL_SCALAPACK_LIBRARIES ${MKL_SCALAPACK_LIBRARY})

  if (NOT TARGET MKL::scalapack)
    add_library(MKL::scalapack INTERFACE IMPORTED GLOBAL)
  endif()

  target_link_libraries(MKL::scalapack
    INTERFACE
      ${MKL_SCALAPACK_LIBRARIES}
  )
endif()
