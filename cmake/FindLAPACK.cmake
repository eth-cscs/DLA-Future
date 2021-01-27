#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# Find LAPACK library
#
# This module finds an installed library that implements the LAPACK linear-algebra interface.
#
# This module sets the following variables:
#  LAPACK_FOUND - set to true if a library implementing the LAPACK interface is found
#
# Following options are allowed (for setting options from code see next section):
#   LAPACK_TYPE - it can be "Compiler" or "Custom"
#     - Compiler (Default): The compiler add the scalapack flag automatically therefore no
#                           extra link line has to be added.
#     - Custom: User can specify include folders and libraries through
#   LAPACK_INCLUDE_DIR - used if SCALAPACK_TYPE=Custom
#       ;-list of include paths
#   LAPACK_LIBRARY - used if SCALAPACK_TYPE=Custom
#       ;-list of {lib name, lib filepath, -Llibrary_folder}
#
# If you want to set options from CMake script you cannot use previous variables, instead you should use:
#   - LAPACK_CUSTOM_TYPE
#   - LAPACK_CUSTOM_INCLUDE_DIR
#   - LAPACK_CUSTOM_LIBRARY
#
# It creates target lapack::lapack

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

### MAIN
set(LAPACK_TYPE_OPTIONS "Compiler" "Custom")
set(LAPACK_TYPE_DEFAULT "Compiler")

# allow to set LAPACK_TYPE from code
if (LAPACK_CUSTOM_TYPE)
  set(LAPACK_TYPE ${LAPACK_CUSTOM_TYPE} CACHE STRING "BLAS/LAPACK type set from script" FORCE)
else()
  set(LAPACK_TYPE ${LAPACK_TYPE_DEFAULT} CACHE STRING "BLAS/LAPACK type setting")
endif()
set_property(CACHE LAPACK_TYPE PROPERTY STRINGS ${LAPACK_TYPE_OPTIONS})
check_valid_option(${LAPACK_TYPE} ${LAPACK_TYPE_OPTIONS})

if(LAPACK_TYPE STREQUAL "Compiler")
  if (NOT LAPACK_CUSTOM_INCLUDE_DIR AND NOT LAPACK_CUSTOM_LIBRARY)
    # reset variables
    set(LAPACK_INCLUDE_DIR "" CACHE STRING "" FORCE)
    set(LAPACK_LIBRARY "" CACHE STRING "" FORCE)
  else()
    message(FATAL_ERROR
      "With LAPACK_TYPE='${LAPACK_TYPE}',
      LAPACK_CUSTOM_INCLUDE_DIR and LAPACK_CUSTOM_LIBRARY must be EMPTY.
      Instead they are set to
      LAPACK_CUSTOM_INCLUDE_DIR='${LAPACK_CUSTOM_INCLUDE_DIR}'
      LAPACK_CUSTOM_LIBRARY='${LAPACK_CUSTOM_LIBRARY}'")
  endif()
elseif(LAPACK_TYPE STREQUAL "Custom")
  # allow to set LAPACK_INCLUDE_DIR from code
  if (LAPACK_CUSTOM_INCLUDE_DIR)
    set(LAPACK_INCLUDE_DIR "${LAPACK_CUSTOM_INCLUDE_DIR}" CACHE STRING
      "BLAS and LAPACK include path for LAPACK_TYPE=Custom (from code)"
      FORCE
    )
  else()
    set(LAPACK_INCLUDE_DIR "" CACHE STRING
      "BLAS and LAPACK include path for LAPACK_TYPE=Custom"
    )
  endif()

  # allow to set LAPACK_LIBRARY from code
  if (LAPACK_CUSTOM_LIBRARY)
    set(LAPACK_LIBRARY "${LAPACK_CUSTOM_LIBRARY}" CACHE STRING
      "BLAS and LAPACK link line for LAPACK_TYPE=Custom (from code)"
      FORCE
    )
  else()
    set(LAPACK_LIBRARY "" CACHE STRING
      "BLAS and LAPACK link line for LAPACK_TYPE=Custom"
    )
  endif()
else()
  message(FATAL_ERROR "Unknown LAPACK type: ${LAPACK_TYPE}")
endif()

mark_as_advanced(
  LAPACK_TYPE
  LAPACK_INCLUDE_DIR
  LAPACK_LIBRARY
)

### Checks
include(CMakePushCheckState)
cmake_push_check_state(RESET)

include(CheckFunctionExists)

set(CMAKE_REQUIRED_INCLUDES ${LAPACK_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARY})

unset(LAPACK_CHECK_BLAS CACHE)
check_function_exists(dgemm_ LAPACK_CHECK_BLAS)
if (NOT LAPACK_CHECK_BLAS)
  message(FATAL_ERROR "BLAS symbol not found with this configuration")
endif()

unset(LAPACK_CHECK CACHE)
check_function_exists(dpotrf_ LAPACK_CHECK)
if (NOT LAPACK_CHECK)
  message(FATAL_ERROR "LAPACK symbol not found with this configuration")
endif()

cmake_pop_check_state()


### Package
if (LAPACK_TYPE STREQUAL "Compiler")
  set(LAPACK_FOUND TRUE)
else()
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(LAPACK DEFAULT_MSG
    LAPACK_LIBRARY
  )
endif()

if (LAPACK_FOUND)
  set(LAPACK_INCLUDE_DIRS ${LAPACK_INCLUDE_DIR})
  set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})

  if (NOT TARGET lapack::lapack)
    add_library(lapack::lapack INTERFACE IMPORTED GLOBAL)
  endif()

  set_target_properties(lapack::lapack
    PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${LAPACK_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}"
  )
endif()
