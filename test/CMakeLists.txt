#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

add_library(DLAF_test INTERFACE)
target_include_directories(DLAF_test
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_library(gtest_mpi_main STATIC src/gtest_mpi_main.cpp)
target_link_libraries(gtest_mpi_main
  PUBLIC
    gtest
    MPI::MPI_CXX
)

add_subdirectory(src)
add_subdirectory(unit)
