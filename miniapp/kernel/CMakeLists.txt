#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

if(DLAF_BUILD_TESTING)
  # TODO they depends on DLAF_TEST exclusively for the createTile method.
  DLAF_addMiniapp(
    miniapp_larft_gemv SOURCES miniapp_larft_gemv.cpp LIBRARIES dlaf.core DLAF_test DLAF_miniapp
  )

  DLAF_addMiniapp(miniapp_laset SOURCES miniapp_laset.cpp LIBRARIES dlaf.core DLAF_test)
endif()
