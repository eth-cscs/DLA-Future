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
  test_multiplication_general
  SOURCES test_multiplication_general.cpp
  LIBRARIES dlaf.multiplication dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_multiplication_hermitian
  SOURCES test_multiplication_hermitian.cpp
  LIBRARIES dlaf.multiplication dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_multiplication_triangular
  SOURCES test_multiplication_triangular.cpp
  LIBRARIES dlaf.multiplication dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)
