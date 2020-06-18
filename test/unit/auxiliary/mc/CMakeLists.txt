#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_norm
  SOURCES test_norm.cpp
  LIBRARIES dlaf.auxiliary dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)
