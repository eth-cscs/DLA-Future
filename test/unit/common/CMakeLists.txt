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
  test_index2d
  SOURCES test_index2d.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)
DLAF_addTest(
  test_size2d
  SOURCES test_size2d.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)
DLAF_addTest(
  test_range2d
  SOURCES test_range2d.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)
DLAF_addTest(
  test_data_descriptor
  SOURCES test_data_descriptor.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)
DLAF_addTest(
  test_pipeline
  SOURCES test_pipeline.cpp
  LIBRARIES dlaf.core
  USE_MAIN PIKA
)
DLAF_addTest(
  test_single_threaded_blas
  SOURCES test_single_threaded_blas.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)
