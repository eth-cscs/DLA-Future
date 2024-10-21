#!/usr/bin/env bash

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# This script helps to plot the results of different benchmarks.

set -eu

####################################################################################################
# Variables to modify:
debug=0
python_venv_path=""
# Path where to find the benchmarks data as the root of z/ and d/ (since they are appended later)
# (list containing several paths if we want to compare data on the same plots)
base_paths=(
    "" \
    "" \
    )
# Path where you want your plotting results
out_path=""
####################################################################################################

if [[ -z $out_path ]] || [[ -z ${base_paths[0]} ]]; then
    echo "You need to set the variables in the beginning of the script"
    exit 1
fi

if [[ ! -z "$python_venv_path" ]]; then
    source $python_venv_path/bin/activate
fi

complex_paths=(${base_paths[@]/%//z})
double_paths=(${base_paths[@]/%//d})
out_path_complex=${out_path/%//z}
out_path_double=${out_path/%//d}

args_base="--distinguish-dir"
args_double="$args_base --out-path ${out_path_double}"
args_complex="$args_base --out-path ${out_path_complex}"

idx=0
for path in "${base_paths[@]}"; do
    args_complex+=" --path ${complex_paths[$idx]}"
    args_double+=" --path ${double_paths[$idx]}"
    idx=$((idx+1))
done

if [[ "$debug" == 1 ]]; then
    BOLD=$(tput bold)
    NORMAL=$(tput sgr0)
    echo "${BOLD}double_paths list:${NORMAL} ${double_paths[@]}"
    echo "${BOLD}complex_paths list:${NORMAL} ${complex_paths[@]}"
    echo "${BOLD}double_args:${NORMAL} $args_double"
    echo "${BOLD}complex_args:${NORMAL} $args_complex"
else
    set -x
    # double
    ./plot_chol_strong.py $args_double &
    ./plot_band2trid_strong.py $args_double &
    ./plot_hegst_strong.py $args_double &
    ./plot_trmm_strong.py $args_double &
    ./plot_bt_band2trid_strong.py $args_double &
    ./plot_evp_strong.py $args_double &
    ./plot_red2band_strong.py $args_double &
    ./plot_trsm_strong.py $args_double &
    ./plot_bt_red2band_strong.py $args_double &
    ./plot_gevp_strong.py $args_double &
    ./plot_tridiag_solver_strong.py $args_double &

    # complex &
    ./plot_chol_strong.py $args_complex &
    ./plot_band2trid_strong.py $args_complex &
    ./plot_hegst_strong.py $args_complex &
    ./plot_trmm_strong.py $args_complex &
    ./plot_bt_band2trid_strong.py $args_complex &
    ./plot_evp_strong.py $args_complex &
    ./plot_red2band_strong.py $args_complex &
    ./plot_trsm_strong.py $args_complex &
    ./plot_bt_red2band_strong.py $args_complex &
    ./plot_gevp_strong.py $args_complex &
    ./plot_tridiag_solver_strong.py $args_complex &

    wait
fi
