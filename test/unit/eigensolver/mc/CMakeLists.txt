#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_reduction_to_band
  SOURCES test_reduction_to_band.cpp
  LIBRARIES dlaf.eigensolver dlaf.factorization dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)
