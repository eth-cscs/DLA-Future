#
# Search variable:
#   MKL_ROOT
#
# Imported target:
#   dlaf::mkl
#
cmake_minimum_required(VERSION 3.12)

# Use the environment variable MKLROOT if MKL_ROOT is not set
if(NOT DEFINED MKL_ROOT)
  set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "MKL's root directory.")
endif()

# Find dependencies
find_package(Threads)

# Determine MKL's library folder
set(_mkl_libpath_suffix "lib/intel64")
if(CMAKE_SIZEOF_VOID_P EQUAL 4) # 32 bit
  set(_mkl_libpath_suffix "lib/ia32")
endif()

if(WIN32)
  string(APPEND _mkl_libpath_suffix "_win")
elseif(APPLE)
  string(APPEND _mkl_libpath_suffix "_mac")
else()
  string(APPEND _mkl_libpath_suffix "_lin")
endif()

# Find MKL header
find_path(
  DLAF_MKL_INCLUDE_DIR mkl.h
    PATHS "${MKL_ROOT}"
          "${MKL_ROOT}/mkl"
    PATH_SUFFIXES include
)
mark_as_advanced(DLAF_MKL_INCLUDE_DIR)

# Find MKL libraries
function(__mkl_find_library _name)
    find_library(
      ${_name} NAMES ${ARGN}
      PATHS "${MKL_ROOT}"
            "${MKL_ROOT}/mkl"
            "${MKL_ROOT}/compiler"
      PATH_SUFFIXES ${_mkl_libpath_suffix}
    )
    mark_as_advanced(${_name})
endfunction()

__mkl_find_library( DLAF_MKL_CORE_LIB       mkl_core       )
__mkl_find_library( DLAF_MKL_INTERFACE_LIB  mkl_intel_lp64 )
__mkl_find_library( DLAF_MKL_SEQ_LIB        mkl_sequential )

# Check if variables are found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  DLAF_MKL REQUIRED_VARS DLAF_MKL_INCLUDE_DIR
                         DLAF_MKL_INTERFACE_LIB
                         DLAF_MKL_SEQ_LIB
                         DLAF_MKL_CORE_LIB
                         Threads_FOUND
)

# Define an imported target
if(DLAF_MKL_FOUND AND NOT TARGET dlaf::mkl)
    set(
      _mkl_libs ${DLAF_MKL_INTERFACE_LIB}
                ${DLAF_MKL_SEQ_LIB}
                ${DLAF_MKL_CORE_LIB}
                Threads::Threads
    )
    add_library(dlaf::mkl INTERFACE IMPORTED)
    set_target_properties(dlaf::mkl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${_mkl_libs}")
endif()
