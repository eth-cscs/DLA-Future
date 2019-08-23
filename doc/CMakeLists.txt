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

macro(transform_to_doxygen_list list)
  string(REPLACE ";" " " ${list} "${${list}}")
endmacro()

set(DOXYGEN_INPUTS
  ${CMAKE_CURRENT_SOURCE_DIR}/mainpage.dox
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
transform_to_doxygen_list(DOXYGEN_INPUTS)

set(DOXYGEN_EXAMPLE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE
)
transform_to_doxygen_list(DOXYGEN_EXAMPLE_PATH)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  @ONLY
)

add_custom_target(doc ALL
  COMMENT "Building Doxygen documentation"
  COMMAND Doxygen::doxygen
)
