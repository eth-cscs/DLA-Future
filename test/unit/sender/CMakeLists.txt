#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_with_temporary_tile
  SOURCES test_with_temporary_tile.cpp
  LIBRARIES dlaf.core
  USE_MAIN PIKA
)
