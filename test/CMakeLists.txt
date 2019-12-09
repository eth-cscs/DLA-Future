#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# ----- MPI
set(MPIEXEC_NUMCORE_FLAG "" CACHE STRING "Flag used by MPI to specify the number of cores per rank for mpiexec. If not empty, you have to specify also the number of cores available per node in MPIEXEC_NUMCORES.")
set(MPIEXEC_NUMCORES "" CACHE STRING "Number of cores available for each MPI rank.")

if ((MPIEXEC_NUMCORE_FLAG OR MPIEXEC_NUMCORES) AND NOT (MPIEXEC_NUMCORE_FLAG AND MPIEXEC_NUMCORES))
  message(FATAL_ERROR "MPIEXEC_NUMCORES and MPIEXEC_NUMCORE_FLAG must be either both sets or both empty.")
endif()


# ----- HPX
set(DLAF_HPXTEST_EXTRA_ARGS "" CACHE STRING "Extra arguments for tests with HPX")


add_library(DLAF_test src/util_mpi.cpp)

target_include_directories(DLAF_test
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(DLAF_test
  PRIVATE
    MPI::MPI_CXX
)

add_subdirectory(src)
add_subdirectory(unit)
