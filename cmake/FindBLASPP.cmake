#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
# Set blas++ library target
#
#  BLASPP_FOUND - set to true if a library implementing the BLASPP interface is found
#
# Following options are required:
#   BLASPP_ROOT - blas++ root directory
#
# It creates target blaspp::blaspp

### Options
set(BLASPP_ROOT "" CACHE STRING "Root directory for blas++")

if(NOT BLASPP_ROOT)
  message(FATAL_ERROR "BLASPP_ROOT unset")
endif()

find_path(BLASPP_INCLUDE_DIR
  blas.hh
  PATHS ${BLASPP_ROOT}/include
)

find_library(BLASPP_LIBRARY
  blaspp
  PATHS ${BLASPP_ROOT}/lib
  NO_DEFAULT_PATH
)

### Package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLASPP DEFAULT_MSG
  BLASPP_LIBRARY
  BLASPP_INCLUDE_DIR
)

if(BLASPP_FOUND)

  set(BLASPP_INCLUDE_DIRS ${BLASPP_INCLUDE_DIR})
  set(BLASPP_LIBRARIES ${BLASPP_LIBRARY})

  add_library(blaspp::blaspp INTERFACE IMPORTED GLOBAL)

  set_target_properties(blaspp::blaspp
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BLASPP_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${BLASPP_LIBRARIES}"
  )

endif()
