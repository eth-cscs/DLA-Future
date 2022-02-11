#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

add_subdirectory(mc)

DLAF_addTest(test_band_to_tridiag
  SOURCES test_band_to_tridiag.cpp
  LIBRARIES dlaf.eigensolver dlaf.factorization dlaf.core
  USE_MAIN PIKA
)

DLAF_addTest(test_bt_band_to_tridiag
  SOURCES test_bt_band_to_tridiag.cpp
  LIBRARIES dlaf.eigensolver dlaf.factorization dlaf.core
  USE_MAIN PIKA
)

DLAF_addTest(test_bt_reduction_to_band
  SOURCES test_bt_reduction_to_band.cpp
  LIBRARIES dlaf.eigensolver dlaf.factorization dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(test_gen_to_std
  SOURCES test_gen_to_std.cpp
  LIBRARIES dlaf.eigensolver dlaf.factorization dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)
