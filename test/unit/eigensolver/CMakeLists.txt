#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

add_subdirectory(mc)

DLAF_addTest(test_band_to_tridiag
  SOURCES test_band_to_tridiag.cpp
  LIBRARIES dlaf.eigensolver dlaf.core
  USE_MAIN HPX
)

DLAF_addTest(test_gen_to_std
  SOURCES test_gen_to_std.cpp
  LIBRARIES dlaf.eigensolver dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)
