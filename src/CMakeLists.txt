#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# Generate version header
find_program(GIT_EXE NAMES git)
string(TIMESTAMP DLAF_TIMESTAMP "%Y-%m-%d %H:%M:%S")
if(DEFINED GIT_EXE AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(COMMAND git rev-parse HEAD
                  OUTPUT_VARIABLE DLAF_SHA
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND git diff --quiet HEAD
                  RESULT_VARIABLE DLAF_GIT_CHANGES
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  ERROR_QUIET)
  # Add a * to git SHA if there are changes
  if (DLAF_GIT_CHANGES EQUAL 1)
    set(DLAF_SHA "${DLAF_SHA} *")
  endif()
else()
  set(DLAF_SHA "N/A : archive")
endif()
configure_file("${PROJECT_SOURCE_DIR}/include/dlaf/version.h.in"
               "${PROJECT_BINARY_DIR}/include/dlaf/version.h"
               @ONLY)

# ----- Options, Flags, Defines, ...

# Check for pretty function support
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("int main() { const char *name = __PRETTY_FUNCTION__; }" is_pretty_function_available)

# Assertions
if (CMAKE_BUILD_TYPE STREQUAL Debug)
  set(DLAF_ASSERT_DEFAULT          ON)
  set(DLAF_ASSERT_MODERATE_DEFAULT ON)
  set(DLAF_ASSERT_HEAVY_DEFAULT    ON)
else()
  set(DLAF_ASSERT_DEFAULT          ON)
  set(DLAF_ASSERT_MODERATE_DEFAULT OFF)
  set(DLAF_ASSERT_HEAVY_DEFAULT    OFF)
endif()

option(DLAF_ASSERT_ENABLE          "Enable low impact assertions"    ${DLAF_ASSERT_DEFAULT})
option(DLAF_ASSERT_MODERATE_ENABLE "Enable medium impact assertions" ${DLAF_ASSERT_MODERATE_DEFAULT})
option(DLAF_ASSERT_HEAVY_ENABLE    "Enable high impact assertions"   ${DLAF_ASSERT_HEAVY_DEFAULT})

# Define DLAF's PUBLIC properties
add_library(dlaf.prop INTERFACE)
target_include_directories(dlaf.prop
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(dlaf.prop
  INTERFACE
    MPI::MPI_CXX
    LAPACK::LAPACK
    pika::pika
    lapackpp
    blaspp
    umpire
    $<TARGET_NAME_IF_EXISTS:dlaf::cublas>
    $<TARGET_NAME_IF_EXISTS:dlaf::cudart>
    $<TARGET_NAME_IF_EXISTS:dlaf::cusolver>
)
target_compile_features(dlaf.prop
  INTERFACE
    cxx_std_17
)
target_compile_definitions(dlaf.prop
  INTERFACE
    $<$<BOOL:${DLAF_ASSERT_ENABLE}>:DLAF_ASSERT_ENABLE>
    $<$<BOOL:${DLAF_ASSERT_MODERATE_ENABLE}>:DLAF_ASSERT_MODERATE_ENABLE>
    $<$<BOOL:${DLAF_ASSERT_HEAVY_ENABLE}>:DLAF_ASSERT_HEAVY_ENABLE>
    DLAF_FUNCTION_NAME=$<IF:$<BOOL:is_pretty_function_available>,__PRETTY_FUNCTION__,__func__>
    $<$<BOOL:${DLAF_WITH_CUDA}>:DLAF_WITH_CUDA>
    $<$<BOOL:${DLAF_WITH_CUDA_MPI_RDMA}>:DLAF_WITH_CUDA_MPI_RDMA>
)

# Define DLAF's CORE library
add_library(dlaf.core
  OBJECT
    communication/communicator_impl.cpp
    communication/communicator.cpp
    communication/communicator_grid.cpp
    communication/datatypes.cpp
    init.cpp
    matrix/distribution.cpp
    matrix/layout_info.cpp
    matrix/tile.cpp
    matrix.cpp
    matrix_mirror.cpp
    memory/memory_view.cpp
    memory/memory_chunk.cpp
    $<$<BOOL:${DLAF_WITH_CUDA}>:cusolver/assert_info.cu>
)
target_compile_options(dlaf.core PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
target_link_libraries(dlaf.core PUBLIC dlaf.prop)
target_add_warnings(dlaf.core)

# Define DLAF's auxiliary library
add_library(dlaf.auxiliary OBJECT auxiliary/norm/mc.cpp)
target_link_libraries(dlaf.auxiliary PUBLIC dlaf.prop)
target_add_warnings(dlaf.auxiliary)

# Define DLAF's eigensolver library
add_library(dlaf.eigensolver OBJECT
  eigensolver/backtransformation/mc.cpp
  eigensolver/band_to_tridiag/mc.cpp
  eigensolver/bt_band_to_tridiag/mc.cpp
  eigensolver/gen_to_std/mc.cpp
  $<$<BOOL:${DLAF_WITH_CUDA}>:eigensolver/gen_to_std/gpu.cpp>
  eigensolver/reduction_to_band/mc.cpp)
target_link_libraries(dlaf.eigensolver PUBLIC dlaf.prop)
target_add_warnings(dlaf.eigensolver)

# Define DLAF's factorization library
add_library(dlaf.factorization OBJECT
  factorization/cholesky/mc.cpp
  $<$<BOOL:${DLAF_WITH_CUDA}>:factorization/cholesky/gpu.cpp>
  factorization/qr/mc.cpp
  $<$<BOOL:${DLAF_WITH_CUDA}>:factorization/qr/gpu.cpp>
  $<$<BOOL:${DLAF_WITH_CUDA}>:factorization/qr/tfactor_kernels.cu>
)
target_compile_options(dlaf.factorization PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
target_link_libraries(dlaf.factorization PUBLIC dlaf.prop)
target_add_warnings(dlaf.factorization)

# Define DLAF's multiplication library
add_library(dlaf.multiplication OBJECT
  multiplication/triangular/mc.cpp
  $<$<BOOL:${DLAF_WITH_CUDA}>:multiplication/triangular/gpu.cpp>)
target_link_libraries(dlaf.multiplication PUBLIC dlaf.prop)
target_add_warnings(dlaf.multiplication)

# Define DLAF's solver library
add_library(dlaf.solver OBJECT
  solver/triangular/mc.cpp
  $<$<BOOL:${DLAF_WITH_CUDA}>:solver/triangular/gpu.cpp>)
target_link_libraries(dlaf.solver PUBLIC dlaf.prop)
target_add_warnings(dlaf.solver)

# Define DLAF's complete library
add_library(DLAF
  $<TARGET_OBJECTS:dlaf.core>
  $<TARGET_OBJECTS:dlaf.auxiliary>
  $<TARGET_OBJECTS:dlaf.eigensolver>
  $<TARGET_OBJECTS:dlaf.factorization>
  $<TARGET_OBJECTS:dlaf.multiplication>
  $<TARGET_OBJECTS:dlaf.solver>
)
target_link_libraries(DLAF PUBLIC dlaf.prop)
target_add_warnings(DLAF)

# ----- DEPLOY
include(GNUInstallDirs)

install(TARGETS
  DLAF dlaf.prop
  EXPORT DLAF-Targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install includes
install(DIRECTORY ../include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install custom FindModules
install(DIRECTORY ../cmake/
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
  FILES_MATCHING PATTERN "Find*.cmake"
  PATTERN "template" EXCLUDE
)

# install version header
install(FILES "${PROJECT_BINARY_DIR}/include/dlaf/version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# ----- CMake INTEGRATION
include(CMakePackageConfigHelpers)

# install targets configuration
install(EXPORT
  DLAF-Targets
  NAMESPACE DLAF::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
)

# Config-file preparation and install
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/../cmake/template/DLAFConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/DLAFConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/DLAFConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
)
