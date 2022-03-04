#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# Loads all scripts available in CMAKE_MODULE_PATH with filename "DLAF_*.cmake"
foreach(module_folder IN LISTS CMAKE_MODULE_PATH)
  file(GLOB cmake_modules "${module_folder}/DLAF_*.cmake")

  foreach(script_filepath IN LISTS cmake_modules)
    get_filename_component(script_name ${script_filepath} NAME_WE)
    include(${script_name})
  endforeach()
endforeach()
