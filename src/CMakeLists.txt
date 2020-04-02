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

target_add_warnings(DLAF)

# ----- Options, Flags, Defines, ...
# DLAF_FUNCTION_NAME
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("int main() { const char *name = __PRETTY_FUNCTION__; }" is_pretty_function_available)

target_compile_definitions(DLAF PUBLIC
  DLAF_FUNCTION_NAME=$<IF:$<BOOL:is_pretty_function_available>,__PRETTY_FUNCTION__,__func__>
)

# Assertions
if (CMAKE_BUILD_TYPE STREQUAL Debug)
  set(DLAF_ASSERT_DEFAULT_LOW  ON)
  set(DLAF_ASSERT_DEFAULT_MED  ON)
  set(DLAF_ASSERT_DEFAULT_HIGH ON)
else()
  set(DLAF_ASSERT_DEFAULT_LOW  ON)
  set(DLAF_ASSERT_DEFAULT_MED  OFF)
  set(DLAF_ASSERT_DEFAULT_HIGH OFF)
endif()

# DLAF_ASSERT_DEFAULT_BUILDTYPE makes persistent the build type for which the current set of flags is configured
# Checking its value, it is possible to reset to flag values when build type is changed
# Build type comparison is case sensitive (Debug != Release != RelWithDebInfo != release)
#
# In order to force a flag configuration, you have to:
# - set DLAF_ASSERT_DEFAULT_BUILDTYPE=${CMAKE_BUILD_TYPE} (in order to bypass this check)
# - set DLAF_ASSERT_ENABLE_* variable(s) as you prefer
if (NOT DLAF_ASSERT_DEFAULT_BUILDTYPE STREQUAL CMAKE_BUILD_TYPE)
  set(DLAF_ASSERT_DEFAULT_BUILDTYPE ${CMAKE_BUILD_TYPE} CACHE INTERNAL "" FORCE)

  set(DLAF_ASSERT_ENABLE_LOW  ${DLAF_ASSERT_DEFAULT_LOW}  CACHE BOOL "Enable low impact assertions"    FORCE)
  set(DLAF_ASSERT_ENABLE_MED  ${DLAF_ASSERT_DEFAULT_MED}  CACHE BOOL "Enable medium impact assertions" FORCE)
  set(DLAF_ASSERT_ENABLE_HIGH ${DLAF_ASSERT_DEFAULT_HIGH} CACHE BOOL "Enable high impact assertions"   FORCE)
endif()

target_compile_definitions(DLAF PUBLIC
  $<$<BOOL:${DLAF_ASSERT_ENABLE_LOW}>:DLAF_ASSERT_ENABLE_LOW>
  $<$<BOOL:${DLAF_ASSERT_ENABLE_MED}>:DLAF_ASSERT_ENABLE_MED>
  $<$<BOOL:${DLAF_ASSERT_ENABLE_HIGH}>:DLAF_ASSERT_ENABLE_HIGH>
)

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
