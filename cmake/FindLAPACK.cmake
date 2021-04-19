#
# CMake recipes
#
# Copyright (c) 2018-2021, ETH Zurich
# BSD 3-Clause License. All rights reserved.
#
# author: Alberto Invernizzi (a.invernizzi@cscs.ch)
#

# Find LAPACK library
#
# LAPACK depends on BLAS and it is up to the user to honor this dependency by specifying
# all the dependencies for the selected LAPACK implementation.
#
# Users can manually specify next variables (even by setting them empty to force use of
# the compiler implicit linking) to control which implementation they want to use:
#   LAPACK_LIBRARY
#       ;-list of {lib name, lib filepath, -Llibrary_folder}
#
# This module sets the following variables:
#   LAPACK_FOUND - set to true if a library implementing the LAPACK interface is found
#
# If LAPACK symbols got found, it creates target LAPACK::LAPACK

macro(_lapack_check_is_working)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)

  include(CheckFunctionExists)

  set(CMAKE_REQUIRED_QUIET TRUE)

  if (NOT LAPACK_LIBRARY STREQUAL "LAPACK_LIBRARIES-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARY})
  endif()

  unset(_LAPACK_CHECK_BLAS CACHE)
  check_function_exists(dgemm_ _LAPACK_CHECK_BLAS)
  if (NOT _LAPACK_CHECK_BLAS)
    message(FATAL_ERROR "BLAS symbol not found with this configuration")
  endif()

  unset(_LAPACK_CHECK CACHE)
  check_function_exists(dpotrf_ _LAPACK_CHECK)
  if (NOT _LAPACK_CHECK)
    message(FATAL_ERROR "LAPACK symbol not found with this configuration")
  endif()

  cmake_pop_check_state()
endmacro()


# Dependencies
set(_DEPS "")

if (LAPACK_LIBRARY STREQUAL "" OR NOT LAPACK_LIBRARY)
  set(LAPACK_LIBRARY "LAPACK_LIBRARIES-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
endif()

mark_as_advanced(
  LAPACK_LIBRARY
)

_lapack_check_is_working()

### Package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACK DEFAULT_MSG
  LAPACK_LIBRARY
  _LAPACK_CHECK
  _LAPACK_CHECK_BLAS
)

# Remove the placeholder
if (LAPACK_LIBRARY STREQUAL "LAPACK_LIBRARIES-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
  set(LAPACK_LIBRARY "")
endif()

if (LAPACK_FOUND)
  if (NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED GLOBAL)
  endif()

  target_link_libraries(LAPACK::LAPACK INTERFACE
    "${LAPACK_LIBRARY}"
    "${_DEPS}"
  )
endif()
