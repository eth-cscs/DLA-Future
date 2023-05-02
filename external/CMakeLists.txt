#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

set(CMAKE_CXX_STANDARD 17)

### GoogleTest
include(FetchContent)

FetchContent_Declare(
  googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.13.0
)
set(BUILD_GMOCK OFF CACHE INTERNAL "")
set(INSTALL_GTEST OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(googletest)
