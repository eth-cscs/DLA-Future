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
  test_band_to_tridiag
  SOURCES test_band_to_tridiag.cpp
  LIBRARIES dlaf.eigensolver dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_bt_band_to_tridiag
  SOURCES test_bt_band_to_tridiag.cpp
  LIBRARIES dlaf.eigensolver dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_tridiag_solver_local
  SOURCES test_tridiag_solver_local.cpp
  LIBRARIES dlaf.tridiagonal_eigensolver dlaf.core
  USE_MAIN PIKA
)

DLAF_addTest(
  test_tridiag_solver_merge
  SOURCES test_tridiag_solver_merge.cpp
  LIBRARIES dlaf.tridiagonal_eigensolver dlaf.core
  USE_MAIN PIKA
)

DLAF_addTest(
  test_eigensolver
  SOURCES test_eigensolver.cpp
  LIBRARIES dlaf.eigensolver dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_gen_eigensolver
  SOURCES test_gen_eigensolver.cpp
  LIBRARIES dlaf.eigensolver dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_reduction_to_band
  SOURCES test_reduction_to_band.cpp
  LIBRARIES dlaf.eigensolver dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_bt_reduction_to_band
  SOURCES test_bt_reduction_to_band.cpp
  LIBRARIES dlaf.eigensolver dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_gen_to_std
  SOURCES test_gen_to_std.cpp
  LIBRARIES dlaf.eigensolver dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_tridiag_solver_rot
  SOURCES test_tridiag_solver_rot.cpp
  LIBRARIES dlaf.tridiagonal_eigensolver dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_tridiag_solver_distributed
  SOURCES test_tridiag_solver_distributed.cpp
  LIBRARIES dlaf.core dlaf.tridiagonal_eigensolver dlaf.multiplication dlaf.permutations
  USE_MAIN MPIPIKA
  MPIRANKS 6
)
