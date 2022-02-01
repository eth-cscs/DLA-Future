#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# Custom GoogleTest MPI Listener
add_library(gtest_mpi_listener OBJECT gtest_mpi_listener.cpp)
target_compile_features(gtest_mpi_listener PUBLIC cxx_std_17)
target_link_libraries(gtest_mpi_listener PUBLIC gtest MPI::MPI_CXX)

add_library(DLAF_gtest_pika_main STATIC gtest_pika_main.cpp)
target_link_libraries(DLAF_gtest_pika_main
  PUBLIC
    gtest
  PRIVATE
    dlaf.core
    pika::pika
)
target_add_warnings(DLAF_gtest_pika_main)

add_library(DLAF_gtest_mpi_main STATIC gtest_mpi_main.cpp)
target_link_libraries(DLAF_gtest_mpi_main
  PUBLIC
    gtest
  PRIVATE
    gtest_mpi_listener
)
target_add_warnings(DLAF_gtest_mpi_main)

add_library(DLAF_gtest_mpipika_main STATIC gtest_mpipika_main.cpp)
target_link_libraries(DLAF_gtest_mpipika_main
  PUBLIC
    gtest
  PRIVATE
    dlaf.core
    gtest_mpi_listener
    pika::pika
)
target_add_warnings(DLAF_gtest_mpipika_main)
