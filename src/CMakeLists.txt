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
  execute_process(
    COMMAND git rev-parse HEAD
    OUTPUT_VARIABLE DLAF_SHA
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND git diff --quiet HEAD
    RESULT_VARIABLE DLAF_GIT_CHANGES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ERROR_QUIET
  )
  # Add a * to git SHA if there are changes
  if(DLAF_GIT_CHANGES EQUAL 1)
    set(DLAF_SHA "${DLAF_SHA} *")
  endif()
else()
  set(DLAF_SHA "N/A : archive")
endif()
configure_file(
  "${PROJECT_SOURCE_DIR}/include/dlaf/version.h.in" "${PROJECT_BINARY_DIR}/include/dlaf/version.h" @ONLY
)

# ----- Options, Flags, Defines, ...

# Check for pretty function support
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "int main() { const char *name = __PRETTY_FUNCTION__; }" is_pretty_function_available
)

# Assertions
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(DLAF_ASSERT_DEFAULT ON)
  set(DLAF_ASSERT_MODERATE_DEFAULT ON)
  set(DLAF_ASSERT_HEAVY_DEFAULT ON)
else()
  set(DLAF_ASSERT_DEFAULT ON)
  set(DLAF_ASSERT_MODERATE_DEFAULT OFF)
  set(DLAF_ASSERT_HEAVY_DEFAULT OFF)
endif()

option(DLAF_ASSERT_ENABLE "Enable low impact assertions" ${DLAF_ASSERT_DEFAULT})
option(DLAF_ASSERT_MODERATE_ENABLE "Enable medium impact assertions" ${DLAF_ASSERT_MODERATE_DEFAULT})
option(DLAF_ASSERT_HEAVY_ENABLE "Enable high impact assertions" ${DLAF_ASSERT_HEAVY_DEFAULT})

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
    DLAF::LAPACK
    pika::pika
    lapackpp
    blaspp
    umpire
    $<$<BOOL:${DLAF_WITH_CUDA}>:CUDA::cublas>
    $<$<BOOL:${DLAF_WITH_CUDA}>:CUDA::cudart>
    $<$<BOOL:${DLAF_WITH_CUDA}>:CUDA::cusolver>
    $<$<BOOL:${DLAF_WITH_HIP}>:roc::hipblas>
    $<$<BOOL:${DLAF_WITH_HIP}>:roc::rocblas>
    $<$<BOOL:${DLAF_WITH_HIP}>:roc::rocsolver>
    $<$<BOOL:${DLAF_WITH_GPU}>:whip::whip>
)
target_compile_definitions(dlaf.prop
  INTERFACE
    $<$<BOOL:${DLAF_ASSERT_ENABLE}>:DLAF_ASSERT_ENABLE>
    $<$<BOOL:${DLAF_ASSERT_MODERATE_ENABLE}>:DLAF_ASSERT_MODERATE_ENABLE>
    $<$<BOOL:${DLAF_ASSERT_HEAVY_ENABLE}>:DLAF_ASSERT_HEAVY_ENABLE>
    DLAF_FUNCTION_NAME=$<IF:$<BOOL:is_pretty_function_available>,__PRETTY_FUNCTION__,__func__>
    $<$<BOOL:${DLAF_WITH_GPU}>:DLAF_WITH_GPU>
    $<$<BOOL:${DLAF_WITH_CUDA}>:DLAF_WITH_CUDA>
    $<$<BOOL:${DLAF_WITH_HIP}>:DLAF_WITH_HIP>
    $<$<BOOL:${DLAF_WITH_HIP}>:ROCM_MATHLIBS_API_USE_HIP_COMPLEX>
    $<$<BOOL:${DLAF_WITH_CUDA_MPI_RDMA}>:DLAF_WITH_CUDA_MPI_RDMA>
)

# Precompiled headers
if(DLAF_WITH_PRECOMPILED_HEADERS)
  # We create two separate targets whose precompiled headers will be reused, one
  # for libraries and one for executables. Separate targets are needed because
  # the compilation flags for executables and libraries may differ.
  add_library(dlaf.pch_lib OBJECT dummy.cpp)
  target_link_libraries(dlaf.pch_lib PRIVATE dlaf.prop)
  target_add_warnings(dlaf.pch_lib)

  add_executable(dlaf.pch_exe dummy.cpp)
  target_link_libraries(dlaf.pch_exe PRIVATE dlaf.prop)
  target_add_warnings(dlaf.pch_exe)

  set(precompiled_headers
      <mpi.h>
      $<$<BOOL:${DLAF_WITH_CUDA}>:<pika/cuda.hpp$<ANGLE-R>>
      <pika/execution.hpp>
      <pika/future.hpp>
      <pika/mutex.hpp>
      <pika/program_options.hpp>
      <pika/runtime.hpp>
      <pika/thread.hpp>
      <pika/unwrap.hpp>
      <blas.hh>
      # We exclude lapack.hh because it pulls in complex.h and defines I as a
      # macro. I is a commonly used e.g. as template parameter names and defining
      # it as a macro breaks compilation. Undefining I for the precompiled header
      # is a bigger hassle than excluding the header since it's a cheap header to
      # compile.
      # <lapack.hh>
      <umpire/Allocator.hpp>
      <umpire/ResourceManager.hpp>
      $<$<BOOL:${DLAF_WITH_CUDA}>:<cublas_v2.h$<ANGLE-R>>
      $<$<BOOL:${DLAF_WITH_CUDA}>:<cuda_runtime.h$<ANGLE-R>>
      $<$<BOOL:${DLAF_WITH_CUDA}>:<cusolverDn.h$<ANGLE-R>>
  )
  target_precompile_headers(dlaf.pch_lib PRIVATE ${precompiled_headers})
  target_precompile_headers(dlaf.pch_exe PRIVATE ${precompiled_headers})
endif()

# Helper function for creating sub-libraries that libDLAF consists of. It
# creates two variants of the library: one object library with the name
# dlaf.${name}_object and one static/shared library with the name dlaf.${name}.
# The object libraries are only meant for creating libDLAF. The static/shared
# libraries are meant for internal use in tests and miniapps.
function(DLAF_addSublibrary name)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SOURCES GPU_SOURCES COMPILE_OPTIONS LIBRARIES)
  cmake_parse_arguments(DLAF_ASL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(object_lib_name dlaf.${name}_object)
  set(lib_name dlaf.${name})

  add_library(${object_lib_name} OBJECT ${DLAF_ASL_SOURCES})
  if(DLAF_WITH_GPU)
    target_sources(${object_lib_name} PRIVATE ${DLAF_ASL_GPU_SOURCES})
    if(DLAF_WITH_HIP)
      set_source_files_properties(${DLAF_ASL_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
    endif()
  endif()

  target_compile_options(${object_lib_name} PRIVATE ${DLAF_ASL_COMPILE_OPTIONS})
  target_link_libraries(${object_lib_name} PRIVATE dlaf.prop)
  target_add_warnings(${object_lib_name})
  DLAF_addPrecompiledHeaders(${object_lib_name})
  add_library(${lib_name} $<TARGET_OBJECTS:${object_lib_name}>)
  target_link_libraries(${lib_name} PUBLIC dlaf.prop)
  target_link_libraries(${lib_name} PUBLIC ${DLAF_ASL_LIBRARIES})
endfunction()

# Define DLAF's CORE library
DLAF_addSublibrary(
  core
  SOURCES communication/communicator_impl.cpp
          communication/communicator.cpp
          communication/communicator_grid.cpp
          communication/datatypes.cpp
          communication/kernels/all_reduce.cpp
          communication/kernels/broadcast.cpp
          communication/kernels/p2p.cpp
          communication/kernels/reduce.cpp
          init.cpp
          matrix/distribution.cpp
          matrix/layout_info.cpp
          matrix/tile.cpp
          matrix.cpp
          matrix_mirror.cpp
          memory/memory_view.cpp
          memory/memory_chunk.cpp
  GPU_SOURCES cusolver/assert_info.cu cusolver/stedc.cu lapack/gpu/lacpy.cu lapack/gpu/laset.cu
  COMPILE_OPTIONS $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
)

# Define DLAF's auxiliary library
DLAF_addSublibrary(auxiliary SOURCES auxiliary/norm/mc.cpp LIBRARIES dlaf.core)

# Define DLAF's eigensolver library
DLAF_addSublibrary(
  eigensolver
  SOURCES eigensolver/band_to_tridiag/mc.cpp
          eigensolver/bt_band_to_tridiag/mc.cpp
          $<$<BOOL:${DLAF_WITH_GPU}>:eigensolver/bt_band_to_tridiag/gpu.cpp>
          eigensolver/bt_reduction_to_band/mc.cpp
          $<$<BOOL:${DLAF_WITH_GPU}>:eigensolver/bt_reduction_to_band/gpu.cpp>
          eigensolver/eigensolver/mc.cpp
          $<$<BOOL:${DLAF_WITH_GPU}>:eigensolver/eigensolver/gpu.cpp>
          eigensolver/gen_eigensolver/mc.cpp
          $<$<BOOL:${DLAF_WITH_GPU}>:eigensolver/gen_eigensolver/gpu.cpp>
          eigensolver/gen_to_std/mc.cpp
          $<$<BOOL:${DLAF_WITH_GPU}>:eigensolver/gen_to_std/gpu.cpp>
          eigensolver/reduction_to_band/mc.cpp
          $<$<BOOL:${DLAF_WITH_GPU}>:eigensolver/reduction_to_band/gpu.cpp>
  LIBRARIES dlaf.tridiagonal_eigensolver dlaf.solver dlaf.factorization dlaf.core
)

# Define DLAF's tridiagonal eigensolver library
DLAF_addSublibrary(
  tridiagonal_eigensolver
  SOURCES eigensolver/tridiag_solver/mc.cpp
          eigensolver/tridiag_solver/kernels.cpp
          $<$<BOOL:${DLAF_WITH_CUDA}>:eigensolver/tridiag_solver/gpu.cpp>
          # TODO: This should be moved to GPU_SOURCES once HIP is supported for the
          # tridiagonal solver.
          $<$<BOOL:${DLAF_WITH_CUDA}>:eigensolver/tridiag_solver/kernels.cu>
  LIBRARIES dlaf.factorization dlaf.multiplication dlaf.permutations dlaf.solver dlaf.core
)

# Define DLAF's factorization library
DLAF_addSublibrary(
  factorization
  SOURCES factorization/cholesky/mc.cpp $<$<BOOL:${DLAF_WITH_GPU}>:factorization/cholesky/gpu.cpp>
          factorization/qr/mc.cpp $<$<BOOL:${DLAF_WITH_GPU}>:factorization/qr/gpu.cpp>
  LIBRARIES dlaf.core
)

# Define DLAF's multiplication library
DLAF_addSublibrary(
  multiplication
  SOURCES multiplication/general/mc.cpp $<$<BOOL:${DLAF_WITH_GPU}>:multiplication/general/gpu.cpp>
          multiplication/triangular/mc.cpp $<$<BOOL:${DLAF_WITH_GPU}>:multiplication/triangular/gpu.cpp>
  LIBRARIES dlaf.core
)

# Define DLAF's permutations library
DLAF_addSublibrary(
  permutations
  SOURCES permutations/general/mc.cpp $<$<BOOL:${DLAF_WITH_GPU}>:permutations/general/gpu.cpp>
  GPU_SOURCES permutations/general/perms.cu
  LIBRARIES dlaf.core
)

# Define DLAF's solver library
DLAF_addSublibrary(
  solver SOURCES solver/triangular/mc.cpp $<$<BOOL:${DLAF_WITH_GPU}>:solver/triangular/gpu.cpp>
  LIBRARIES dlaf.core
)

# Define DLAF's complete library
add_library(
  DLAF
  $<TARGET_OBJECTS:dlaf.core_object>
  $<TARGET_OBJECTS:dlaf.auxiliary_object>
  $<TARGET_OBJECTS:dlaf.eigensolver_object>
  $<TARGET_OBJECTS:dlaf.tridiagonal_eigensolver_object>
  $<TARGET_OBJECTS:dlaf.factorization_object>
  $<TARGET_OBJECTS:dlaf.multiplication_object>
  $<TARGET_OBJECTS:dlaf.permutations_object>
  $<TARGET_OBJECTS:dlaf.solver_object>
)
target_link_libraries(DLAF PUBLIC dlaf.prop)
target_add_warnings(DLAF)

# ----- DEPLOY
include(GNUInstallDirs)

install(
  TARGETS DLAF dlaf.prop
  EXPORT DLAF-Targets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install includes
install(DIRECTORY ../include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install custom FindModules
install(
  DIRECTORY ../cmake/
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
  FILES_MATCHING
  PATTERN "Find*.cmake"
  PATTERN "template" EXCLUDE
)

# install version header
install(FILES "${PROJECT_BINARY_DIR}/include/dlaf/version.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# ----- CMake INTEGRATION
include(CMakePackageConfigHelpers)

# install targets configuration
install(EXPORT DLAF-Targets NAMESPACE DLAF::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
)

# Config-file preparation and install
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/../cmake/template/DLAFConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/DLAFConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DLAFConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
)
