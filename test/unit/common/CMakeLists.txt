#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_index2d SOURCES test_index2d.cpp USE_MAIN PLAIN)
DLAF_addTest(test_size2d SOURCES test_size2d.cpp USE_MAIN PLAIN)
DLAF_addTest(test_range2d SOURCES test_range2d.cpp USE_MAIN PLAIN)
DLAF_addTest(test_data_descriptor SOURCES test_data_descriptor.cpp USE_MAIN PLAIN)

DLAF_addTest(test_pipeline SOURCES test_pipeline.cpp USE_MAIN HPX)
DLAF_addTest(test_pool SOURCES test_pool.cpp USE_MAIN HPX)
