#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(
  test_eigensolver_c_api
  SOURCES test_eigensolver_c_api.cpp test_eigensolver_c_api_wrapper.c
  LIBRARIES dlaf.c_api
  USE_MAIN CAPI
  MPIRANKS 6
)

DLAF_addTest(
  test_gen_eigensolver_c_api
  SOURCES test_gen_eigensolver_c_api.cpp test_gen_eigensolver_c_api_wrapper.c
          ../factorization/test_cholesky_c_api_wrapper.c
  LIBRARIES dlaf.c_api
  USE_MAIN CAPI
  MPIRANKS 6
)
