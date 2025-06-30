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
  test_inverse_from_cholesky_factor
  SOURCES test_inverse_from_cholesky_factor.cpp
  LIBRARIES dlaf.inverse dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_triangular_inverse
  SOURCES test_triangular_inverse.cpp
  LIBRARIES dlaf.inverse dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)
