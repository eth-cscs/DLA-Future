#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_util_math SOURCES test_util_math.cpp USE_GTEST_MAIN)
DLAF_addTest(test_tile SOURCES test_tile.cpp LIBRARIES PRIVATE DLAF_gtest_hpx_main)

add_subdirectory(common)
add_subdirectory(memory)
add_subdirectory(communicator)
