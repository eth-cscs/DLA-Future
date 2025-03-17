#
# CMake recipes
#
# Copyright (c) ETH Zurich
# BSD 3-Clause License. All rights reserved.
#
# author: Alberto Invernizzi (a.invernizzi@cscs.ch)
#

# dlaf-no-license-check

# Find LAPACK library
#
# LAPACK depends on BLAS and it is up to the user to honor this dependency by specifying
# all the dependencies for the selected LAPACK implementation.
#
# Users can manually specify next variables (even by setting them empty to force use of
# the compiler implicit linking) to control which implementation they want to use:
#   DLAF_LAPACK_LIBRARY
#       ;-list of {lib name, lib filepath, -Llibrary_folder}
#
#   DLAF_LAPACK_INCLUDE_DIR
#       ;-list of include folders
#
# This module sets the following variables:
#   DLAF_LAPACK_FOUND - set to true if a library implementing the LAPACK interface is found
#
# If LAPACK symbols got found, it creates target DLAF::LAPACK

macro(_lapack_check_is_working)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)

  include(CheckFunctionExists)

  set(CMAKE_REQUIRED_QUIET TRUE)

  if(NOT DLAF_LAPACK_LIBRARY STREQUAL "DLAF_LAPACK_LIBRARIES-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${DLAF_LAPACK_LIBRARY})
  endif()

  unset(_DLAF_LAPACK_CHECK_BLAS CACHE)
  check_function_exists(dgemm_ _DLAF_LAPACK_CHECK_BLAS)
  if(NOT _DLAF_LAPACK_CHECK_BLAS)
    message(FATAL_ERROR "BLAS symbol not found with this configuration")
  endif()

  unset(_DLAF_LAPACK_CHECK CACHE)
  check_function_exists(dpotrf_ _DLAF_LAPACK_CHECK)
  if(NOT _DLAF_LAPACK_CHECK)
    message(FATAL_ERROR "LAPACK symbol not found with this configuration")
  endif()

  cmake_pop_check_state()
endmacro()

# Dependencies
set(_DEPS "")

if(NOT DLAF_LAPACK_LIBRARY)
  set(DLAF_LAPACK_LIBRARY "DLAF_LAPACK_LIBRARIES-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
endif()

mark_as_advanced(DLAF_LAPACK_LIBRARY)

_lapack_check_is_working()

if(NOT DLAF_LAPACK_INCLUDE_DIR)
  set(DLAF_LAPACK_INCLUDE_DIR "DLAF_LAPACK_INCLUDE_DIR-PLACEHOLDER-FOR-EMPTY-INCLUDE-DIR")
endif()

mark_as_advanced(DLAF_LAPACK_INCLUDE_DIR)

### Package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  DLAF_LAPACK DEFAULT_MSG DLAF_LAPACK_LIBRARY DLAF_LAPACK_INCLUDE_DIR _DLAF_LAPACK_CHECK
  _DLAF_LAPACK_CHECK_BLAS
)

# Remove the placeholders
if(DLAF_LAPACK_LIBRARY STREQUAL "DLAF_LAPACK_LIBRARIES-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
  set(DLAF_LAPACK_LIBRARY "")
endif()

if(DLAF_LAPACK_INCLUDE_DIR STREQUAL "DLAF_LAPACK_INCLUDE_DIR-PLACEHOLDER-FOR-EMPTY-INCLUDE-DIR")
  set(DLAF_LAPACK_INCLUDE_DIR "")
endif()

if(DLAF_LAPACK_FOUND)
  if(NOT TARGET DLAF::LAPACK)
    add_library(DLAF::LAPACK INTERFACE IMPORTED GLOBAL)
  endif()

  target_link_libraries(DLAF::LAPACK INTERFACE "${DLAF_LAPACK_LIBRARY}" "${_DEPS}")
  target_include_directories(DLAF::LAPACK INTERFACE "${DLAF_LAPACK_INCLUDE_DIR}")
endif()
