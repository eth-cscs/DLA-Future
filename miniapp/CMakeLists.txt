#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

add_executable(miniapp_cholesky miniapp_cholesky.cpp)
target_link_libraries(miniapp_cholesky PRIVATE dlaf.core dlaf.auxiliary dlaf.factorization)
target_add_warnings(miniapp_cholesky)

add_executable(miniapp_tsgemm miniapp_tsgemm.cpp)
target_link_libraries(miniapp_tsgemm PRIVATE dlaf.core)
target_add_warnings(miniapp_tsgemm)

if (DLAF_BUILD_TESTING)
  # TODO it depends on DLAF_TEST exclusively for the check part (uses CHECK_MATRIX_NEAR)
  add_executable(miniapp_triangular_solver miniapp_triangular_solver.cpp)
  target_link_libraries(miniapp_triangular_solver PRIVATE DLAF)
  target_link_libraries(miniapp_triangular_solver PRIVATE DLAF_test)
  target_add_warnings(miniapp_triangular_solver)
endif()

add_executable(miniapp_reduction_to_band miniapp_reduction_to_band.cpp)
target_link_libraries(miniapp_reduction_to_band PRIVATE DLAF)
target_add_warnings(miniapp_reduction_to_band)

if(DLAF_WITH_CUDA)
  add_executable(miniapp_cublas miniapp_cublas.cpp)
  target_link_libraries(miniapp_cublas PRIVATE DLAF)
  target_add_warnings(miniapp_cublas)
endif()
