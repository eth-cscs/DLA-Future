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

run_dir = "~/ws/runs/weak"

# Note: job time is computed as time0 + sqrt(nodes) * time
time0 = 120  # minutes
time = 0  # minutes
nruns = 5
nodes_arr = [1, 2, 4, 8, 16]

rpn = 1
m_szs_d = [10240, 20480, 30097, 40960]
mb_szs_d = 512
m_szs_z = [10240, 20480]
mb_szs_z = 512

extra_flags = "--dlaf:bt-band-to-tridiag-hh-apply-group-size=128"

approx = 512  # the sizes used in weak scaling are chosen to be the nearest multiple of approx.

parser = argparse.ArgumentParser(description="Run weak scaling benchmarks.")
parser.add_argument(
    "--debug",
    help="Don't submit jobs, only create job scripts instead.",
    action="store_true",
)
args = parser.parse_args()

debug = args.debug


def createAndSubmitRun(run_dir, nodes_arr, dtype, **kwargs):
    if dtype == "d":
        m_szs = m_szs_d
        mb_szs = mb_szs_d
        run_dir += "/d"
    elif dtype == "z":
        m_szs = m_szs_z
        mb_szs = mb_szs_z
        run_dir += "/z"
    else:
        raise RuntimeError(f"Invalid type specified {dtype}")

    full_kwargs = kwargs.copy()
    full_kwargs["lib"] = "dlaf"
    full_kwargs["build_dir"] = dlafpath
    full_kwargs["approx"] = approx
    full_kwargs["nruns"] = nruns
    full_kwargs["dtype"] = dtype

    run = mp.WeakScaling(system, "DLAF_test_weak", "job_dlaf", nodes_arr, time0, time)

    run.add(
        mp.chol,
        params={"rpn": rpn, "mb_sz": mb_szs},
        weak_params={"m_sz": m_szs},
        **full_kwargs,
    )
    run.add(
        mp.gen2std,
        params={"rpn": rpn, "mb_sz": mb_szs},
        weak_params={"m_sz": m_szs},
        **full_kwargs,
    )
    run.add(
        mp.red2band,
        params={"rpn": rpn, "mb_sz": mb_szs, "band": 128},
        weak_params={"m_sz": m_szs},
        **full_kwargs,
    )
    run.add(
        mp.band2trid,
        params={"rpn": rpn, "mb_sz": mb_szs, "band": 128},
        weak_params={"m_sz": m_szs},
        **full_kwargs,
    )
    run.add(
        mp.trid_evp,
        params={"rpn": rpn, "mb_sz": mb_szs},
        weak_params={"m_sz": m_szs},
        **full_kwargs,
    )
    run.add(
        mp.bt_band2trid,
        params={"rpn": rpn, "mb_sz": mb_szs, "band": 128, "n_sz": None},
        weak_params={"m_sz": m_szs},
        **full_kwargs,
    )
    run.add(
        mp.bt_red2band,
        params={"rpn": rpn, "mb_sz": mb_szs, "band": 128, "n_sz": None},
        weak_params={"m_sz": m_szs},
        **full_kwargs,
    )
    run.add(
        mp.trsm,
        params={"rpn": rpn, "mb_sz": mb_szs, "n_sz": None},
        weak_params={"m_sz": m_szs},
        **full_kwargs,
    )

    fullsolver_args = full_kwargs.copy()
    fullsolver_args["extra_flags"] = fullsolver_args.get("extra_flags", "") + " --check=last"

    run.add(
        mp.evp,
        params={"rpn": rpn, "mb_sz": mb_szs, "min_band": None},
        weak_params={"m_sz": m_szs},
        **fullsolver_args,
    )
    run.add(
        mp.gevp,
        params={"rpn": rpn, "mb_sz": mb_szs, "min_band": None},
        weak_params={"m_sz": m_szs},
        **fullsolver_args,
    )

    run.submit(run_dir, debug=debug)


# actual benchmark
createAndSubmitRun(run_dir, nodes_arr, "d", extra_flags=extra_flags)
createAndSubmitRun(run_dir, nodes_arr, "z", extra_flags=extra_flags)
