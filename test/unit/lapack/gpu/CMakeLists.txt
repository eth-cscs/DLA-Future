#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

if(DLAF_WITH_GPU)
  DLAF_addTest(
    test_lacpy
    SOURCES test_lacpy.cpp
    LIBRARIES dlaf.core
    USE_MAIN PLAIN
  )
endif()
