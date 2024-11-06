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

time = 120  # minutes
nruns = 5
nodes_arr = [1, 2, 4, 8, 16]

m_szs_d = [10240, 20480, 30097, 40960]
mb_szs_d = 512
m_szs_z = [10240, 20480]
mb_szs_z = 512

extra_flags = "--dlaf:bt-band-to-tridiag-hh-apply-group-size=128"

parser = argparse.ArgumentParser(description="Run strong scaling benchmarks.")
parser.add_argument(
    "--miniapps",
    default="<path_to_dlaf_miniapp_dir>",
    help="Path to DLA-Future miniapps directory.",
    type=str,
)
parser.add_argument(
    "--references",
    default="<path_to_h5_refs>",
    help="Path to reference matrices in HDF5 format.",
    type=str,
)
parser.add_argument(
    "--rundir",
    default="~/ws/runs/strong",
    help="Directory where to run the benchmarks.",
    type=str,
)
parser.add_argument(
    "--system",
    default="daint-gpu",
    help="System to run the benchmarks on.",
    type=str,
)
parser.add_argument(
    "--debug",
    help="Don't submit jobs, only create job scripts instead.",
    action="store_true",
)
parser.add_argument(
    "--percent-evals",
    help="Apply back transformation to a percentage of eigenvectors.",
    default=None,
    type=float,
)
parser.add_argument(
    "--rpns",
    help="Ranks per node.",
    default=1,
    type=int,
)
args = parser.parse_args()

system = systems.cscs[args.system]
dlafpath = args.miniapps
matrixrefpath = args.references
run_dir = args.rundir
debug = args.debug
rpn = args.rpns


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

    fullbt_kwargs = full_kwargs.copy()
    if args.percent_evals is not None:
        fullbt_kwargs["extra_flags"] = fullbt_kwargs.get("extra_flags", "") + f" --percent-evals={args.percent_evals}"

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
        **fullbt_kwargs,
    )
    run.add(
        mp.bt_red2band,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "band": 128, "n_sz": None},
        **fullbt_kwargs,
    )
    # TODO: Partial trsm for Cholesky
    run.add(
        mp.trsm,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "n_sz": None},
        **full_kwargs,
    )

    fullsolver_args = fullbt_kwargs.copy()
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
