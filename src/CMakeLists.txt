#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

add_library(DLAF
  communication/communicator_impl.cpp
  communication/communicator.cpp
  communication/communicator_grid.cpp
  communication/datatypes.cpp
  factorization/mc.cpp
  matrix/distribution.cpp
  matrix/layout_info.cpp
  matrix.cpp
  memory/memory_view.cpp
  tile.cpp
  solver/mc.cpp
)

target_include_directories(DLAF
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
    ${HPX_INCLUDE_DIRS}
  PRIVATE
    include/
)

target_link_libraries(DLAF
  PUBLIC
    MPI::MPI_CXX
    ${HPX_LIBRARIES}
    ${LAPACK_TARGET}
    lapackpp
    blaspp
)

target_compile_features(DLAF PUBLIC cxx_std_14)

target_compile_definitions(DLAF PUBLIC DLAF_SOURCE_LOCATION=__func__)

target_add_warnings(DLAF)

# ----- DEPLOY
include(GNUInstallDirs)

install(TARGETS
  DLAF
  EXPORT DLAF-Targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install includes
install(DIRECTORY ../include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install custom FindModules
install(DIRECTORY ../cmake/
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  FILES_MATCHING PATTERN "Find*.cmake"
  PATTERN "template" EXCLUDE
)

# ----- CMake INTEGRATION
include(CMakePackageConfigHelpers)

# install targets configuration
install(EXPORT
  DLAF-Targets
  NAMESPACE DLAF::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
)

# Config-file preparation and install
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/../cmake/template/DLAFConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/DLAFConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/DLAFConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
)
