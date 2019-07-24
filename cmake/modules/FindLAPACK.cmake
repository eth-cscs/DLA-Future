#
# NS3C
#
# Copyright (c) 2018-2019, ETH Zurich
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
# Following options are allowed:
#   LAPACK_TYPE - it can be "Compiler" or "Custom"
#     - Compiler (Default): The compiler add the scalapack flag automatically therefore no
#                           extra link line has to be added.
#     - Custom: User can specify include folders and libraries through
#   LAPACK_INCLUDE_DIR - used if SCALAPACK_TYPE=Custom
#       ;-list of include paths
#   LAPACK_LIBRARY - used if SCALAPACK_TYPE=Custom
#       ;-list of {lib name, lib filepath, -Llibrary_folder}
#
# It creates target lapack::lapack

### Options
set(LAPACK_TYPE "Compiler" CACHE STRING "BLAS/LAPACK type setting")
set_property(CACHE LAPACK_TYPE PROPERTY STRINGS "Compiler" "Custom")

set(LAPACK_INCLUDE_DIR "" CACHE STRING "BLAS and LAPACK include path for DLA_LAPACK_TYPE = Custom")
set(LAPACK_LIBRARY "" CACHE STRING "BLAS and LAPACK link line for DLA_LAPACK_TYPE = Custom")


if(LAPACK_TYPE STREQUAL "Compiler")
  # reset variables
  set(LAPACK_INCLUDE_DIR "" CACHE STRING "" FORCE)
  set(LAPACK_LIBRARY "" CACHE STRING "" FORCE)
elseif(LAPACK_TYPE STREQUAL "Custom")
  # nothing to do
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
    LAPACK_INCLUDE_DIR
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
