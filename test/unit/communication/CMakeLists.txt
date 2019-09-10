#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_communicator
  SOURCES test_communicator.cpp
  MPIRANKS 4
  USE_GTEST_MAIN
)

DLAF_addTest(test_communicator_grid
  SOURCES test_communicator_grid.cpp
  MPIRANKS 4
  USE_GTEST_MAIN
)
