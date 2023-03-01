#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

cmake_minimum_required(VERSION 3.22)
project(DLAF-miniapps)

list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(DLAF REQUIRED)
find_package(pika 0.11.0 REQUIRED)

cmake_dependent_option(DLAF_MINIAPP_CSV_OUTPUT "Output CSV data in supported miniapps" OFF "DLAF_BUILD_MINIAPPS" OFF)
include(DLAF_AddMiniapp)
include(DLAF_AddTargetWarnings)

add_library(DLAF_miniapp INTERFACE)
target_link_libraries(DLAF_miniapp INTERFACE DLAF::DLAF pika::pika)
target_include_directories(DLAF_miniapp INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

DLAF_addMiniapp(miniapp_cholesky SOURCES miniapp_cholesky.cpp)

DLAF_addMiniapp(miniapp_gen_to_std SOURCES miniapp_gen_to_std.cpp)

DLAF_addMiniapp(miniapp_reduction_to_band SOURCES miniapp_reduction_to_band.cpp)

DLAF_addMiniapp(miniapp_band_to_tridiag SOURCES miniapp_band_to_tridiag.cpp)

DLAF_addMiniapp(miniapp_tridiag_solver SOURCES miniapp_tridiag_solver.cpp)

DLAF_addMiniapp(miniapp_bt_band_to_tridiag SOURCES miniapp_bt_band_to_tridiag.cpp)

DLAF_addMiniapp(miniapp_bt_reduction_to_band SOURCES miniapp_bt_reduction_to_band.cpp)

DLAF_addMiniapp(miniapp_triangular_solver SOURCES miniapp_triangular_solver.cpp)

DLAF_addMiniapp(miniapp_eigensolver SOURCES miniapp_eigensolver.cpp)

DLAF_addMiniapp(miniapp_gen_eigensolver SOURCES miniapp_gen_eigensolver.cpp)

add_subdirectory(kernel)
