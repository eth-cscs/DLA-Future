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
  test_grid_c_api
  SOURCES test_grid_c_api.cpp test_grid_c_api_wrapper.c
  LIBRARIES dlaf.c_api
  USE_MAIN MPIPIKA
  MPIRANKS 6
  CATEGORY CAPI
)
