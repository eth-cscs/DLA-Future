#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_compute_t_factor
  SOURCES test_compute_t_factor.cpp
  LIBRARIES dlaf.factorization dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)
