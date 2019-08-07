#
# NS3C
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
# Set blas++ and lapack++ library targets
#
#  BLASPP_FOUND - set to true if a library implementing the BLASPP interface is found
#  LAPACKPP_FOUND - set to true if a library implementing the LAPACKPP interface is found
#
# Following options are required:
#   BLASPP_DIR - blas++ root directory
#   LAPACKPP_DIR - lapack++ root directory
#
# It creates targets blaspp::blaspp lapackpp::lapackpp

### Options
set(BLASPP_DIR "" CACHE STRING "Root directory for blas++")
set(LAPACKPP_DIR "" CACHE STRING "Root directory for lapack++")

if(BLASPP_DIR STREQUAL "")
  message(FATAL_ERROR "BLASPP_DIR unset")
endif()

if(LAPACKPP_DIR STREQUAL "")
  message(FATAL_ERROR "LAPACKPP_DIR unset")
endif()


### Package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLASPP DEFAULT_MSG
  BLASPP_DIR
)
find_package_handle_standard_args(LAPACKPP DEFAULT_MSG
  LAPACKPP_DIR
)

if(BLASPP_FOUND AND LAPACKPP_FOUND)
  set(BLASPP_INCLUDE_DIR ${BLASPP_DIR}/include)
  set(BLASPP_LIBRARY ${BLASPP_DIR}/lib/libblaspp.so)

  set(LAPACKPP_INCLUDE_DIR ${LAPACKPP_DIR}/include)
  set(LAPACKPP_LIBRARY ${LAPACKPP_DIR}/lib/liblapackpp.so)

  add_library(blaspp::blaspp INTERFACE IMPORTED GLOBAL)
  add_library(lapackpp::lapackpp INTERFACE IMPORTED GLOBAL)

  set_target_properties(blaspp::blaspp
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BLASPP_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${BLASPP_LIBRARY}"
  )
  set_target_properties(lapackpp::lapackpp
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LAPACKPP_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${LAPACKPP_LIBRARY}"
  )

endif()
