#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

add_executable(miniapp_cholesky miniapp_cholesky.cpp)
target_link_libraries(miniapp_cholesky PRIVATE DLAF)
target_add_warnings(miniapp_cholesky)

add_executable(miniapp_triangular_solver miniapp_triangular_solver.cpp)
target_link_libraries(miniapp_triangular_solver PRIVATE DLAF DLAF_test)
target_add_warnings(miniapp_triangular_solver)
