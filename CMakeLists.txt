cmake_minimum_required(VERSION 3.12.4)

# ---------------------------------------------------------------------------
# PROJECT
# ---------------------------------------------------------------------------
project(NS3C C CXX)

set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
endif()

# ---------------------------------------------------------------------------
# CMake configurations
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# ---------------------------------------------------------------------------
# Libraries configuration
# ---------------------------------------------------------------------------

# ----- MPI
find_package(MPI REQUIRED)

# ----- LAPACK/SCALAPACK
set(NS3C_WITH_MKL OFF CACHE BOOL "Enable MKL as provider for LAPACK")
if (NS3C_WITH_MKL)
  message(WARNING "MKL Threading is fixed to sequential!")
  set(MKL_THREADING "Sequential")
  find_package(MKL REQUIRED)

  add_library(lapack::lapack ALIAS MKL::lapack)
else()
  find_package(LAPACK REQUIRED)
endif()

# ----- HPX
find_package(HPX REQUIRED)

# ----- EXTERNAL
add_subdirectory(external)

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
set(NS3C_WITH_TEST BOOL "ON" "Enables tests")
if (NS3C_WITH_TEST)
  include(CTest)
  enable_testing()
  
  add_subdirectory(test)
endif()