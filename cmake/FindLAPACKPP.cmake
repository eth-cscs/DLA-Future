#
# NS3C
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
# Set lapack++ library target
#
#  LAPACKPP_FOUND - set to true if a library implementing the LAPACKPP interface is found
#
# Following options are required:
#   LAPACKPP_ROOT - lapack++ root directory
#
# It creates targets lapackpp::lapackpp

### Options
set(LAPACKPP_ROOT "" CACHE STRING "Root directory for lapack++")

if(NOT LAPACKPP_ROOT)
  message(FATAL_ERROR "LAPACKPP_ROOT unset")
endif()

find_path(LAPACKPP_INCLUDE_DIR
  lapack.hh
  PATHS ${LAPACKPP_ROOT}/include
)

find_library(LAPACKPP_LIBRARY
  lapackpp
  PATHS ${LAPACKPP_ROOT}/lib
  NO_DEFAULT_PATH
)

### Package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKPP DEFAULT_MSG
  LAPACKPP_LIBRARY
  LAPACKPP_INCLUDE_DIR
)

if(LAPACKPP_FOUND)

  set(LAPACKPP_INCLUDE_DIRS ${LAPACKPP_INCLUDE_DIR})
  set(LAPACKPP_LIBRARIES ${LAPACKPP_LIBRARY})

  add_library(lapackpp::lapackpp INTERFACE IMPORTED GLOBAL)

  set_target_properties(lapackpp::lapackpp
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LAPACKPP_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${LAPACKPP_LIBRARIES}"
  )

endif()
