#! /usr/bin/env bash

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

FILES_CHANGED=`git diff --name-only origin/master`
TAB_FOUND=0

for FILE in $FILES_CHANGED
do
  if [[ ! -f $FILE ]]
  then
    continue
  fi

  case $FILE in
    *.cpp|*.h|*.h.in|*.tpp|*.cu)
      clang-format-18 -i --style=file $FILE
      # The following is needed for regions in which clang-format is disabled.
      # Note: clang-format removes trailing spaces even in disabled regions.
      # Check if tab are present.
      egrep -Hn $'\t' $FILE && TAB_FOUND=1 || true
      ;;
    *.cmake|*CMakeLists.txt|*.cmake.in)
      cmake-format -i $FILE
      # Check if tab are present.
      egrep -Hn $'\t' $FILE && TAB_FOUND=1 || true
      ;;
    *.pdf|*.hdf5|*.jpg|*.png|*.ppt|*.pptx|*.ipe)
      # Exclude some binary files types,
      # others can be excluded as needed.
      ;;
    *)
      # Remove trailing whitespaces.
      grep -q "\s\+$" $FILE && sed -i "s/\s\+$//g" $FILE || true

      # Count number of trailing newlines at the end of the file.
      # For performance reason we limit the search (max is 32).
      n_newlines=`tail -c 32 $FILE | xxd -p -c 32 | egrep -o "(0a)+$" | grep -o 0a | wc -l`

      # Add or remove trailing newlines if needed.
      if [[ $n_newlines -eq 0 ]]
      then
        echo "" >> $FILE
      elif [[ $n_newlines -gt 1 ]]
      then
        truncate -s -$(($n_newlines-1)) $FILE
      fi
      ;;
  esac
done
# Fail if there are tabs in source files.
if [ $TAB_FOUND -eq 1 ]
then
  echo "Error: Tabs have been found in source files"
  false
else
  true
fi
