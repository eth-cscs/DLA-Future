#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

add_library(DLAF_gtest_hpx_main STATIC gtest_hpx_main.cpp)
target_include_directories(DLAF_gtest_hpx_main PRIVATE ${HPX_INCLUDE_DIRS})
target_link_libraries(DLAF_gtest_hpx_main
  PUBLIC
    gtest
  PRIVATE
    ${HPX_LIBRARIES}
)

add_library(DLAF_gtest_mpi_main STATIC gtest_mpi_main.cpp)
target_link_libraries(DLAF_gtest_mpi_main
  PUBLIC
    gtest
  PRIVATE
    MPI::MPI_CXX
)
