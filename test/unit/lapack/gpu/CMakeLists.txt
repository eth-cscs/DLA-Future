#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

if (DLAF_WITH_CUDA)
  DLAF_addTest(test_lacpy
    SOURCES test_lacpy.cpp
    LIBRARIES dlaf.core
    USE_MAIN PLAIN
  )

  DLAF_addTest(test_larft
    SOURCES test_larft.cpp
    LIBRARIES dlaf.core
    USE_MAIN PLAIN
  )
endif()
