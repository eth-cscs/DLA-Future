#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

### GoogleTest
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0

  # Patch for TestPartResult operator<<
  # Fixes missing filename on exceptions causing segmentation fault
  PATCH_COMMAND git cherry-pick -n 5395345ca4f0c596110188688ed990e0de5a181c -m 1
)

FetchContent_GetProperties(googletest)
if (NOT googletest_POPULATED)
  FetchContent_Populate(googletest)

  # set default values for options
  option(BUILD_GMOCK "Build GoogleTest GMock" OFF)
  option(BUILD_GTEST "Build GoogleTest GTest" ON)

  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)

  # keep project cache clean
  mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )
endif()
