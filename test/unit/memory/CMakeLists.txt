#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_memory_chunk
  SOURCES test_memory_chunk.cpp
  USE_MAIN PLAIN
  LIBRARIES dlaf.core
)
DLAF_addTest(test_memory_view
  SOURCES test_memory_view.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)
