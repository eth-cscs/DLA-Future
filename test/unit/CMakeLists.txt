#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_types
  SOURCES test_types.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(test_util_math
  SOURCES test_util_math.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(test_blas_tile
  SOURCES test_blas_tile.cpp
  LIBRARIES dlaf.core
  USE_MAIN HPX
)
DLAF_addTest(test_lapack_tile
  SOURCES test_lapack_tile.cpp
  LIBRARIES dlaf.core
  USE_MAIN HPX
)
DLAF_addTest(test_tile
  SOURCES test_tile.cpp
  LIBRARIES dlaf.core
  USE_MAIN HPX
)

DLAF_addTest(test_matrix
  SOURCES test_matrix.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)

# Generic libraries
add_subdirectory(common)
add_subdirectory(communication)
add_subdirectory(matrix)
add_subdirectory(memory)

# Operations
add_subdirectory(auxiliary)
add_subdirectory(factorization)
add_subdirectory(solver)
