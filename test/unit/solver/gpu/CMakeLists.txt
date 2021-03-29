#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

if(DLAF_WITH_CUDA)
  DLAF_addTest(test_triangular_gpu
    SOURCES test_triangular.cpp
    LIBRARIES dlaf.solver dlaf.core
    USE_MAIN MPIHPX
    MPIRANKS 6
  )
endif()
