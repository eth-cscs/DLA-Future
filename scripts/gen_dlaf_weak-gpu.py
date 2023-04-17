#!/usr/bin/env python3

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# This file is an example on how to use the miniapp module.
# Please do not add gen scripts used for benchmarks into the source repository,
# they should be kept with the result produced.

import argparse
import miniapps as mp
import systems

system = systems.cscs["daint-gpu"]

dlafpath = "<path_to_dlaf_build_dir>"

run_dir = f"~/ws/runs"


nodes_arr = [1, 2, 4, 8, 16]
rpn = 1
m_szs = [10240, 20480, 30097, 40960]
mb_szs = 1024

nruns = 5
approx = 512  # the sizes used in weak scaling are chosen to be the nearest multiple of approx.

# Note: job time is computed as time0 + sqrt(nodes) * time
time0 = 120  # minutes
time = 0  # minutes


parser = argparse.ArgumentParser(description="Run weak scaling benchmarks.")
parser.add_argument(
    "--debug",
    help="Don't submit jobs, only create job scripts instead.",
    action="store_true",
)
args = parser.parse_args()

debug = args.debug


run = mp.WeakScaling(system, "DLAF_test_weak", "job_dlaf", nodes_arr, time0, time)

run.add(
    mp.chol,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs},
    {"m_sz": m_szs},
    approx,
    nruns,
)
run.add(
    mp.gen2std,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs},
    {"m_sz": m_szs},
    approx,
    nruns,
)
run.add(
    mp.red2band,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs, "band": 128},
    {"m_sz": m_szs},
    approx,
    nruns,
)
run.add(
    mp.band2trid,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs, "band": 128},
    {"m_sz": m_szs},
    approx,
    nruns,
)
run.add(
    mp.trid_evp,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs},
    {"m_sz": m_szs},
    approx,
    nruns,
)
run.add(
    mp.bt_band2trid,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs, "band": 128, "n_sz": None},
    {"m_sz": m_szs},
    approx,
    nruns,
)
run.add(
    mp.bt_red2band,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs, "band": 128, "n_sz": None},
    {"m_sz": m_szs},
    approx,
    nruns,
)
run.add(
    mp.trsm,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs, "n_sz": None},
    {"m_sz": m_szs},
    approx,
    nruns,
)

run.add(
    mp.evp,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs, "min_band": None},
    {"m_sz": m_szs},
    approx,
    nruns,
)
run.add(
    mp.gevp,
    "dlaf",
    dlafpath,
    {"rpn": rpn, "mb_sz": mb_szs, "min_band": None},
    {"m_sz": m_szs},
    approx,
    nruns,
)
run.submit(run_dir, debug=debug)
