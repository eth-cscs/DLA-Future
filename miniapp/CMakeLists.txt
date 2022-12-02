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
target_include_directories(DLAF_miniapp INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

DLAF_addMiniapp(miniapp_cholesky SOURCES miniapp_cholesky.cpp LIBRARIES DLAF)

if(DLAF_BUILD_TESTING)
  # TODO it depends on DLAF_TEST exclusively for the check part (uses CHECK_MATRIX_NEAR)
  DLAF_addMiniapp(
    miniapp_triangular_solver SOURCES miniapp_triangular_solver.cpp LIBRARIES DLAF DLAF_test
  )
endif()

DLAF_addMiniapp(miniapp_gen_to_std SOURCES miniapp_gen_to_std.cpp LIBRARIES DLAF)

DLAF_addMiniapp(miniapp_reduction_to_band SOURCES miniapp_reduction_to_band.cpp LIBRARIES DLAF)

DLAF_addMiniapp(miniapp_band_to_tridiag SOURCES miniapp_band_to_tridiag.cpp LIBRARIES DLAF)

# Note:
# HIP is not yet supported. Disabling this miniapp for HIP means that we don't have also the MC one...
# ...but it is a shortcut in the meanwhile we get HIP support.
DLAF_addMiniapp(miniapp_eigensolver SOURCES miniapp_eigensolver.cpp LIBRARIES DLAF)
DLAF_addMiniapp(miniapp_gen_eigensolver SOURCES miniapp_gen_eigensolver.cpp LIBRARIES DLAF)

DLAF_addMiniapp(miniapp_bt_band_to_tridiag SOURCES miniapp_bt_band_to_tridiag.cpp LIBRARIES DLAF)

add_subdirectory(kernel)
