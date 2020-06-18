#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_layout_info
  SOURCES test_layout_info.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)
DLAF_addTest(test_util_distribution
  SOURCES test_util_distribution.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)
DLAF_addTest(test_distribution
  SOURCES test_distribution.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
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
