#
# NS3C
#
# Copyright (c) 2018-2019, ETH Zurich
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
  GIT_TAG        release-1.8.0
)

FetchContent_GetProperties(googletest)
if (NOT googletest_POPULATED)
  FetchContent_Populate(googletest)

  # set default values for options
  option(BUILD_GMOCK "Build GoogleTest GMock" OFF)
  option(BUILD_GTEST "Build GoogleTest GTest" ON)

  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()
