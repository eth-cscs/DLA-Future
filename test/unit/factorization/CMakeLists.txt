#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_cholesky
  SOURCES test_cholesky.cpp
  LIBRARIES dlaf.factorization dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(test_compute_t_factor
  SOURCES test_compute_t_factor.cpp
  LIBRARIES dlaf.factorization dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)
