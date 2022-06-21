#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# DLAF_addMiniapp(miniapp_target_name
#   SOURCES <source1> [<source2> ...]
#   [COMPILE_DEFINITIONS <arguments for target_compile_definitions>]
#   [INCLUDE_DIRS <arguments for target_include_directories>]
#   [LIBRARIES <arguments for target_link_libraries>]
# )
#
# At least one source file has to be specified, while other parameters are optional.
#
# COMPILE_DEFINITIONS, INCLUDE_DIRS and LIBRARIES are passed to respective cmake wrappers, so it is
# possible to specify PRIVATE/INTERFACE/PUBLIC modifiers.
#
# e.g.
#
# DLAF_addMiniapp(example_miniapp
#   SOURCE main.cpp
#   LIBRARIES
#     PRIVATE
#       boost::boost
#       include/
# )

function(DLAF_addMiniapp miniapp_target_name)
  set(options "")
  set(oneValueArgs MPIRANKS USE_MAIN)
  set(multiValueArgs SOURCES COMPILE_DEFINITIONS INCLUDE_DIRS LIBRARIES ARGUMENTS)
  cmake_parse_arguments(DLAF_AM "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  ### Checks
  if(DLAF_AM_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown arguments ${DLAF_AM_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT DLAF_AM_SOURCES)
    message(FATAL_ERROR "No sources specified for this miniapp")
  endif()

  ### Miniapp executable target
  add_executable(${miniapp_target_name} ${DLAF_AM_SOURCES})
  target_link_libraries(${miniapp_target_name} PRIVATE DLAF_miniapp ${DLAF_AM_LIBRARIES})
  target_compile_definitions(${miniapp_target_name} PRIVATE ${DLAF_AM_COMPILE_DEFINITIONS})
  target_include_directories(${miniapp_target_name} PRIVATE ${DLAF_AM_INCLUDE_DIRS})
  target_add_warnings(${miniapp_target_name})
  DLAF_addPrecompiledHeaders(${miniapp_target_name})

  ### DEPLOY
  include(GNUInstallDirs)

  set(DLAF_INSTALL_MINIAPPS
      ON
      CACHE BOOL "If miniapps are built, it controls if they will be installed"
  )
  if(DLAF_INSTALL_MINIAPPS)
    install(TARGETS ${miniapp_target_name}
                    # EXPORT DLAF-miniapps
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()
endfunction()
