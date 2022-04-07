#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

add_library(DLAF_miniapp INTERFACE)
target_link_libraries(DLAF_miniapp INTERFACE DLAF pika::pika)
target_include_directories(DLAF_miniapp
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_executable(miniapp_cholesky miniapp_cholesky.cpp)
target_link_libraries(miniapp_cholesky PRIVATE dlaf.core dlaf.auxiliary dlaf.factorization)
target_link_libraries(miniapp_cholesky PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_cholesky)
DLAF_addPrecompiledHeaders(miniapp_cholesky)

if (DLAF_BUILD_TESTING)
  # TODO it depends on DLAF_TEST exclusively for the check part (uses CHECK_MATRIX_NEAR)
  add_executable(miniapp_triangular_solver miniapp_triangular_solver.cpp)
  target_link_libraries(miniapp_triangular_solver PRIVATE DLAF)
  target_link_libraries(miniapp_triangular_solver PRIVATE DLAF_test)
  target_link_libraries(miniapp_triangular_solver PRIVATE DLAF_miniapp)
  target_add_warnings(miniapp_triangular_solver)
  DLAF_addPrecompiledHeaders(miniapp_triangular_solver)
endif()

add_executable(miniapp_gen_to_std miniapp_gen_to_std.cpp)
target_link_libraries(miniapp_gen_to_std PRIVATE DLAF)
target_link_libraries(miniapp_gen_to_std PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_gen_to_std)
DLAF_addPrecompiledHeaders(miniapp_gen_to_std)

add_executable(miniapp_reduction_to_band miniapp_reduction_to_band.cpp)
target_link_libraries(miniapp_reduction_to_band PRIVATE DLAF)
target_link_libraries(miniapp_reduction_to_band PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_reduction_to_band)
DLAF_addPrecompiledHeaders(miniapp_reduction_to_band)

add_executable(miniapp_eigensolver miniapp_eigensolver.cpp)
target_link_libraries(miniapp_eigensolver PRIVATE DLAF)
target_link_libraries(miniapp_eigensolver PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_eigensolver)
DLAF_addPrecompiledHeaders(miniapp_eigensolver)

add_executable(miniapp_gen_eigensolver miniapp_gen_eigensolver.cpp)
target_link_libraries(miniapp_gen_eigensolver PRIVATE DLAF)
target_link_libraries(miniapp_gen_eigensolver PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_gen_eigensolver)
DLAF_addPrecompiledHeaders(miniapp_gen_eigensolver)

if(DLAF_WITH_CUDA)
  add_executable(miniapp_cublas miniapp_cublas.cpp)
  target_link_libraries(miniapp_cublas PRIVATE DLAF)
  target_add_warnings(miniapp_cublas)
  DLAF_addPrecompiledHeaders(miniapp_cublas)
endif()

add_subdirectory(kernel)
