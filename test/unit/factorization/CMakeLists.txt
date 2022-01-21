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

DLAF_addTest(test_cholesky
  SOURCES test_cholesky.cpp
  LIBRARIES dlaf.factorization dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)
