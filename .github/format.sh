#! /usr/bin/env bash

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# ----------------------------------------------------------------
# To use this script as a hook, symlink it to your git hooks as follows
# (note that the extra ../../ in the path is because git runs the hook
# from the .git/hooks directory, so the symlink has to be redirected)
# cd <project root (primary worktree if using git worktree dir)>
# ln -s -f ../../.github/format.sh .git/hooks/pre-commit
# ----------------------------------------------------------------

SCRIPT=`dirname $0`
if [ "$SCRIPT" == ".git/hooks" ] ; then
  GIT_HOOK=true
  FILES_CHANGED=`git diff --cached --name-only --diff-filter=ACMRT`
  cxxfiles=()
  cmakefiles=()
  red=$(tput setaf 1)
  green=$(tput setaf 2)
  yellow=$(tput setaf 3)
  blue=$(tput setaf 4)
  normal=$(tput sgr0)
else
  FILES_CHANGED=`git diff --name-only origin/master`
fi

CLANG_FORMAT_VERSION=clang-format-15
TAB_FOUND=0

for FILE in $FILES_CHANGED
do
  if [[ ! -f $FILE ]]
  then
    continue
  fi

  case $FILE in
    *.cpp|*.h|*.h.in|*.tpp|*.cu)
      if [ "$GIT_HOOK" = true ] ; then
        if ! cmp -s <(git show :$FILE) <(git show :$FILE|$CLANG_FORMAT_VERSION -style=file); then
          cxxfiles+=("$FILE")
        fi
      else
        $CLANG_FORMAT_VERSION -i --style=file $FILE
        # The following is needed for regions in which clang-format is disabled.
        # Note: clang-format removes trailing spaces even in disabled regions.
        # Check if tab are present.
        egrep -Hn $'\t' $FILE && TAB_FOUND=1 || true
      fi
      ;;
    *.cmake|*CMakeLists.txt|*.cmake.in)
      if [ "$GIT_HOOK" = true ] ; then
        tmpfile=$(mktemp /tmp/cmake-check.XXXXXX)
        git show :${file} > $tmpfile
        cmake-format -c $(pwd)/.cmake-format.py -i $tmpfile
        if ! cmp -s <(git show :${file}) <(cat $tmpfile); then
          cmakefiles+=("${file}")
        fi
        rm $tmpfile
      else
        cmake-format -i $FILE
        # Check if tab are present.
        egrep -Hn $'\t' $FILE && TAB_FOUND=1 || true
      fi
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

if [ "$GIT_HOOK" = true ] ; then
  returncode=0
  full_list=
  if [ -n "${cxxfiles}" ]; then
    printf "# ${blue}clang-format ${red}error pre-commit${normal} : To fix run the following (use git commit ${yellow}--no-verify${normal} to bypass)\n"
    for f in "${cxxfiles[@]}" ; do
      rel=$(realpath --relative-to "./$GIT_PREFIX" $f)
      printf "$CLANG_FORMAT_VERSION -i %s\n" "$rel"
      full_list="${rel} ${full_list}"
    done
    returncode=1
  fi

  if [ -n "${cmakefiles}" ]; then
    printf "# ${green}cmake-format ${red}error pre-commit${normal} : To fix run the following (use git commit ${yellow}--no-verify${normal} to bypass)\n"
    for f in "${cmakefiles[@]}" ; do
      rel=$(realpath --relative-to "./$GIT_PREFIX" $f)
      printf "cmake-format -i %s\n" "$rel"
      full_list="${rel} ${full_list}"
    done
    returncode=1
  fi

  if [ ! -z "$full_list" ]; then
    printf "\n# ${red}To commit the corrected files, run\n${normal}\ngit add ${full_list}\n"
  fi
  exit $returncode
else
  # Fail if there are tabs in source files.
  if [ $TAB_FOUND -eq 1 ]
  then
    echo "Error: Tabs have been found in source files"
    false
  else
    true
  fi
fi
