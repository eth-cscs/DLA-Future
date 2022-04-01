#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

if (DLAF_BUILD_TESTING)
  # TODO they depends on DLAF_TEST exclusively for the createTile method.
  add_executable(miniapp_laset miniapp_laset.cpp)
  target_link_libraries(miniapp_laset PRIVATE dlaf.core)
  target_link_libraries(miniapp_laset PRIVATE DLAF_test)
  target_link_libraries(miniapp_laset PRIVATE DLAF_miniapp)
  target_add_warnings(miniapp_laset)
endif()
