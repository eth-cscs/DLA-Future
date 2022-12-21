#!/usr/bin/env python3

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
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

system = systems.cscs["daint-mc"]

dlafpath = "<path_to_dlaf>"

run_dir = f"~/ws/runs"

time = 400  # minutes
nruns = 5
nodes_arr = [1, 2, 4]

parser = argparse.ArgumentParser(description="Run strong scaling benchmarks.")
parser.add_argument(
    "--debug",
    help="Don't submit jobs, only create job scripts instead.",
    action="store_true",
)
args = parser.parse_args()

debug = args.debug

run = mp.StrongScaling(system, "DLAF_test_strong", "job_dlaf", nodes_arr, time)
run.add(
    mp.chol,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512},
    nruns,
)
run.add(
    mp.gen2std,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512},
    nruns,
)
run.add(
    mp.red2band,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512, "band": 128},
    nruns,
)
run.add(
    mp.band2trid,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512, "band": 128},
    nruns,
)
run.add(
    mp.trid_evp,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512},
    nruns,
)
run.add(
    mp.bt_band2trid,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512, "band": 128, "n_sz": None},
    nruns,
)
run.add(
    mp.bt_red2band,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512, "band": 128, "n_sz": None},
    nruns,
)
run.add(
    mp.trsm,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512, "n_sz": None},
    nruns,
)

run.add(
    mp.evp,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512, "band": None},
    nruns,
)
run.add(
    mp.gevp,
    "dlaf",
    dlafpath,
    {"rpn": 2, "m_sz": 10240, "mb_sz": 512, "band": None},
    nruns,
)
run.submit(run_dir, debug=debug)
