#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(
  test_triangular
  SOURCES test_triangular.cpp
  LIBRARIES dlaf.solver dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)
