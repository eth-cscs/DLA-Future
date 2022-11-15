#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(
  test_permutations_local
  SOURCES test_permutations_local.cpp
  LIBRARIES dlaf.permutations dlaf.core
  USE_MAIN PIKA
)

DLAF_addTest(
  test_permutations_distributed
  SOURCES test_permutations_distributed.cpp
  LIBRARIES dlaf.permutations dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)
