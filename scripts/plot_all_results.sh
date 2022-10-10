#!/usr/bin/env bash

set -uxo pipefail

dlaf_root=$APPS_SRC/DLA-Future/scripts

[[ -d cholesky_strong ]] && (cd cholesky_strong && "${dlaf_root}/plot_chol_strong.py" --path .)  &
[[ -d cholesky_weak   ]] && (cd cholesky_weak   && "${dlaf_root}/plot_chol_weak.py" --path .)    &
[[ -d trsm_strong     ]] && (cd trsm_strong     && "${dlaf_root}/plot_trsm_strong.py" --path .)  &
[[ -d trsm_weak       ]] && (cd trsm_weak       && "${dlaf_root}/plot_trsm_weak.py" --path .)    &
[[ -d gen2std_strong  ]] && (cd gen2std_strong  && "${dlaf_root}/plot_hegst_strong.py" --path .) &
[[ -d gen2std_weak    ]] && (cd gen2std_weak    && "${dlaf_root}/plot_hegst_weak.py" --path .)   &

wait
