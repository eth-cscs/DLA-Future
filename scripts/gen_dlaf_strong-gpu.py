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

run_dir = f"~/ws/runs/strong"

time = 400  # minutes
nruns = 5
nodes_arr = [1, 2, 4, 8, 16]

rpn = 1
m_szs = [10240, 20480, 30097, 40960]
mb_szs = 1024


parser = argparse.ArgumentParser(description="Run strong scaling benchmarks.")
parser.add_argument(
    "--debug",
    help="Don't submit jobs, only create job scripts instead.",
    action="store_true",
)
args = parser.parse_args()

debug = args.debug


def createAndSubmitRun(run_dir, nodes_arr, **kwargs):
    run = mp.StrongScaling(system, "DLAF_test_strong", "job_dlaf", nodes_arr, time)

    run.add(
        mp.chol,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs},
        nruns, **kwargs,
    )
    run.add(
        mp.gen2std,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs},
        nruns, **kwargs,
    )
    run.add(
        mp.red2band,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "band": 128},
        nruns, **kwargs,
    )
    run.add(
        mp.band2trid,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "band": 128},
        nruns, **kwargs,
    )
    run.add(
        mp.trid_evp,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs},
        nruns, **kwargs,
    )
    run.add(
        mp.bt_band2trid,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "band": 128, "n_sz": None},
        nruns, **kwargs,
    )
    run.add(
        mp.bt_red2band,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "band": 128, "n_sz": None},
        nruns, **kwargs,
    )
    run.add(
        mp.trsm,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "n_sz": None},
        nruns, **kwargs,
    )

    run.add(
        mp.evp,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "min_band": None},
        nruns, **kwargs,
    )
    run.add(
        mp.gevp,
        "dlaf",
        dlafpath,
        {"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "min_band": None},
        nruns, **kwargs,
    )

    run.submit(run_dir, debug=debug)


# actual benchmark
createAndSubmitRun(
        run_dir,
        nodes_arr)

# additional benchmark collecting "local" implementation results in <run_dir>/local sub-directory
createAndSubmitRun(
        run_dir + "-local",
        [1 / rpn],
        extra_flags="--local")
