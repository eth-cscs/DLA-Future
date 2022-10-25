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
  test_norm
  SOURCES test_norm.cpp
  LIBRARIES dlaf.auxiliary dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

target_compile_options(test_broadcast_panel PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wno-error=array-bounds>)
