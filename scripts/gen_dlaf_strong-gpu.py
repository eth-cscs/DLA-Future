#!/usr/bin/env python3

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
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

system = systems.cscs["todi"]

dlafpath = "/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/dla-future-git.202408-develop_0.6.0-iqm3lnzbffriwwwpyd3helfsfsb5oybc/bin"
matrixrefpath = ""
run_dir = ""

time = 12 * 60  # minutes
nruns = 10
nodes_arr = [0.25, 0.5, 1, 2, 4, 8, 16, 32]

rpn = 4
m_szs_d = [10240, 20480, 30097, 40960]
mb_szs_d = [512, 1024]
m_szs_z = [10240, 20480]
mb_szs_z = [512, 1024]

extra_flags = "--pika:threads=64 --nwarmups=0"

parser = argparse.ArgumentParser(description="Run strong scaling benchmarks.")
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
    full_kwargs["miniapp_dir"] = dlafpath
    full_kwargs["nruns"] = nruns
    full_kwargs["dtype"] = dtype

    run = mp.StrongScaling(system, "DLAF_test_strong", "job_dlaf", nodes_arr, time)

    run.add(
        mp.chol,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs},
        **full_kwargs,
    )
    run.add(
        mp.gen2std,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs},
        **full_kwargs,
    )
    run.add(
        mp.red2band,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "band": 128},
        **full_kwargs,
    )
    run.add(
        mp.band2trid,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "band": 128},
        **full_kwargs,
    )
    run.add(
        mp.trid_evp,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs},
        **full_kwargs,
    )
    run.add(
        mp.bt_band2trid,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "band": 128, "n_sz": None},
        **full_kwargs,
    )
    run.add(
        mp.bt_red2band,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "band": 128, "n_sz": None},
        **full_kwargs,
    )
    run.add(
        mp.trsm,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "n_sz": None},
        **full_kwargs,
    )

    fullsolver_args = full_kwargs.copy()
    fullsolver_args["extra_flags"] = fullsolver_args.get("extra_flags", "") + " --check=last"

    run.add(
        mp.evp,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "min_band": None},
        **fullsolver_args,
    )
    run.add(
        mp.gevp,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "min_band": None},
        **fullsolver_args,
    )

    run.submit(run_dir, debug=debug)

    run = mp.StrongScaling(system, "DLAF_test_strong", "job_dlaf-norandom", nodes_arr, time)
    for m_sz in m_szs:
        trid_kwargs = full_kwargs.copy()
        trid_kwargs["suffix"] = "fromfile"
        trid_kwargs["extra_flags"] = (
            trid_kwargs.get("extra_flags", "") + f" --input-file={matrixrefpath}/trid-ref-{m_sz}.h5"
        )

        run.add(
            mp.trid_evp,
            params={"rpn": rpn, "m_sz": m_sz, "mb_sz": mb_szs},
            **trid_kwargs,
        )
    run.submit(run_dir, debug=debug)


# actual benchmark
createAndSubmitRun(run_dir, nodes_arr, "d", extra_flags=extra_flags)
createAndSubmitRun(run_dir, nodes_arr, "z", extra_flags=extra_flags)

# additional benchmark collecting "local" implementation results in <run_dir>-local directory
createAndSubmitRun(run_dir + "-local", [1 / rpn], "d", extra_flags=extra_flags + " --local")
createAndSubmitRun(run_dir + "-local", [1 / rpn], "z", extra_flags=extra_flags + " --local")
