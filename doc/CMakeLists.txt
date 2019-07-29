#
# NS3C
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

find_package(Doxygen REQUIRED)

set(DOXYGEN_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  @ONLY
)

add_custom_target(doc ALL
  COMMENT "Building Doxygen documentation"
  COMMAND Doxygen::doxygen
)
