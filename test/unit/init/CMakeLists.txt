#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2020-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_init
  SOURCES test_init.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)
