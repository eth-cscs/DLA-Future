#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_tile
  SOURCES test_tile.cpp
  LIBRARIES dlaf.core
  USE_MAIN HPX
)

DLAF_addTest(test_layout_info
  SOURCES test_layout_info.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(test_distribution
  SOURCES test_distribution.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(test_util_distribution
  SOURCES test_util_distribution.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(test_matrix
  SOURCES test_matrix.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)

DLAF_addTest(test_matrix_local
  SOURCES test_matrix_local.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)

DLAF_addTest(test_matrix_mirror
  SOURCES test_matrix_mirror.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)

DLAF_addTest(test_matrix_view
  SOURCES test_matrix_view.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)

DLAF_addTest(test_util_matrix
  SOURCES test_util_matrix.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)

DLAF_addTest(test_matrix_output
  SOURCES test_matrix_output.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIHPX
  MPIRANKS 6
)
