#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(
  test_tile
  SOURCES test_tile.cpp
  LIBRARIES dlaf.core
  USE_MAIN PIKA
)

DLAF_addTest(
  test_col_major_layout
  SOURCES test_col_major_layout.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(
  test_distribution
  SOURCES test_distribution.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(
  test_util_distribution
  SOURCES test_util_distribution.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(
  test_distribution_extensions
  SOURCES test_distribution_extensions.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(
  test_matrix
  SOURCES test_matrix.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_matrix_local
  SOURCES test_matrix_local.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_matrix_mirror
  SOURCES test_matrix_mirror.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_util_matrix
  SOURCES test_util_matrix.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_matrix_output
  SOURCES test_matrix_output.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_matrix_ref
  SOURCES test_matrix_ref.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_panel
  SOURCES test_panel.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_retiled_matrix
  SOURCES test_retiled_matrix.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_retiled_matrix_ref
  SOURCES test_retiled_matrix_ref.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_views
  SOURCES test_views.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

DLAF_addTest(
  test_copy
  SOURCES test_copy.cpp
  LIBRARIES dlaf.core
  USE_MAIN MPIPIKA
  MPIRANKS 6
)

if(DLAF_WITH_HDF5)
  DLAF_addTest(
    test_matrix_hdf5
    SOURCES test_matrix_hdf5.cpp
    LIBRARIES dlaf.core
    USE_MAIN MPIPIKA
    MPIRANKS 2
  )
endif()
