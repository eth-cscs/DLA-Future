#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

pkg_search_module(_DLAF_SCALAPACK scalapack)

find_library(
  DLAF_SCALAPACK_LIBRARY NAME scalapack
  HINTS ${_DLAF_SCALAPACK_LIBRARY_DIRS}
        ENV
        SCALAPACKROOT
        SCALAPACK_ROOT
        SCALAPACK_PREFIX
        SCALAPACK_DIR
        SCALAPACKDIR
        /usr
  PATH_SUFFIXES lib
)

find_package_handle_standard_args(DLAF_SCALAPACK DEFAULT_MSG DLAF_SCALAPACK_LIBRARY)

mark_as_advanced(DLAF_SCALAPACK_LIBRARY)
mark_as_advanced(DLAF_SCALAPACK_INCLUDE_DIR)

if(NOT TARGET DLAF::SCALAPACK)
  add_library(DLAF::SCALAPACK INTERFACE IMPORTED GLOBAL)
endif()

target_link_libraries(DLAF::SCALAPACK INTERFACE "${DLAF_SCALAPACK_LIBRARY}")
target_include_directories(DLAF::SCALAPACK INTERFACE "${DLAF_SCALAPACK_INCLUDE_DIR}")
