#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

add_library(DLAF_test INTERFACE)

target_include_directories(DLAF_test INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(DLAF_test INTERFACE gtest)

add_subdirectory(header)
add_subdirectory(src)
add_subdirectory(unit)
