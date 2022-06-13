#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)

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
install(TARGETS miniapp_cholesky RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (DLAF_BUILD_TESTING)
  # TODO it depends on DLAF_TEST exclusively for the check part (uses CHECK_MATRIX_NEAR)
  add_executable(miniapp_triangular_solver miniapp_triangular_solver.cpp)
  target_link_libraries(miniapp_triangular_solver PRIVATE DLAF)
  target_link_libraries(miniapp_triangular_solver PRIVATE DLAF_test)
  target_link_libraries(miniapp_triangular_solver PRIVATE DLAF_miniapp)
  target_add_warnings(miniapp_triangular_solver)
  DLAF_addPrecompiledHeaders(miniapp_triangular_solver)
  install(TARGETS miniapp_triangular_solver RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_executable(miniapp_gen_to_std miniapp_gen_to_std.cpp)
target_link_libraries(miniapp_gen_to_std PRIVATE DLAF)
target_link_libraries(miniapp_gen_to_std PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_gen_to_std)
DLAF_addPrecompiledHeaders(miniapp_gen_to_std)
install(TARGETS miniapp_gen_to_std RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(miniapp_reduction_to_band miniapp_reduction_to_band.cpp)
target_link_libraries(miniapp_reduction_to_band PRIVATE DLAF)
target_link_libraries(miniapp_reduction_to_band PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_reduction_to_band)
DLAF_addPrecompiledHeaders(miniapp_reduction_to_band)
install(TARGETS miniapp_reduction_to_band RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(miniapp_band_to_tridiag miniapp_band_to_tridiag.cpp)
target_link_libraries(miniapp_band_to_tridiag PRIVATE DLAF)
target_link_libraries(miniapp_band_to_tridiag PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_band_to_tridiag)
DLAF_addPrecompiledHeaders(miniapp_band_to_tridiag)
install(TARGETS miniapp_band_to_tridiag RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(miniapp_eigensolver miniapp_eigensolver.cpp)
target_link_libraries(miniapp_eigensolver PRIVATE DLAF)
target_link_libraries(miniapp_eigensolver PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_eigensolver)
DLAF_addPrecompiledHeaders(miniapp_eigensolver)
install(TARGETS miniapp_eigensolver RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(miniapp_gen_eigensolver miniapp_gen_eigensolver.cpp)
target_link_libraries(miniapp_gen_eigensolver PRIVATE DLAF)
target_link_libraries(miniapp_gen_eigensolver PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_gen_eigensolver)
DLAF_addPrecompiledHeaders(miniapp_gen_eigensolver)
install(TARGETS miniapp_gen_eigensolver RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(miniapp_bt_band_to_tridiag miniapp_bt_band_to_tridiag.cpp)
target_link_libraries(miniapp_bt_band_to_tridiag PRIVATE DLAF)
target_link_libraries(miniapp_bt_band_to_tridiag PRIVATE DLAF_miniapp)
target_add_warnings(miniapp_bt_band_to_tridiag)
DLAF_addPrecompiledHeaders(miniapp_bt_band_to_tridiag)

add_subdirectory(kernel)
