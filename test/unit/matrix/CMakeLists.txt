#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(test_layout_info SOURCES test_layout_info.cpp USE_MAIN PLAIN)
DLAF_addTest(test_util_distribution SOURCES test_util_distribution.cpp USE_MAIN PLAIN)
DLAF_addTest(test_distribution SOURCES test_distribution.cpp USE_MAIN PLAIN)
DLAF_addTest(test_util_matrix SOURCES test_util_matrix.cpp USE_MAIN HPX)
