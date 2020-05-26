#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

find_package(Doxygen REQUIRED)

configure_file(Doxyfile.in Doxyfile)

add_custom_target(doc ALL
  COMMENT "Building Doxygen documentation"
  COMMAND Doxygen::doxygen
)

# INSTALL
include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
)
