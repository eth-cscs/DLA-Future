#!/usr/bin/env python3

# This file is an example on how to use the miniapp module.
# Please do not add gen scripts used for benchmarks into the source repository,
# they should be kept with the result produced.

import argparse
import miniapps as mp
import systems
import math

system = systems.cscs["daint-mc"]

libpaths = {
    "dlaf": "<path_to_dlaf>",
    "dplasma": "<path_to_dplasma>",
    "slate": "<path_to_slate>",
    "scalapack-libsci": "<path_to_libsci_miniapp>",
    "scalapack-mkl": "<path_to_mkl_miniapp>",
}

run_dir = f"~/ws/runs_w"

time0 = 20  # minutes
time = 5  # minutes
# Note: job time is computed as time0 + sqrt(nodes) * time

nruns = 10
nodes_arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

approx = 512  # the sizes used in weak scaling are chosen to be the nearest multiple of approx.

parser = argparse.ArgumentParser(description="Run weak scaling benchmarks.")
parser.add_argument(
    "--debug",
    help="Don't submit jobs, only create job scripts instead.",
    action="store_true",
)
parser.add_argument(
    "--libs",
    help="Run miniapps for these libraries.",
    nargs="+",
    default=["scalapack-mkl", "scalapack-libsci", "dlaf", "slate", "dplasma"],
    choices=list(libpaths.keys()),
)
args = parser.parse_args()

debug = args.debug

run_mkl = "scalapack-mkl" in args.libs
run_libsci = "scalapack-libsci" in args.libs
run_dlaf = "dlaf" in args.libs
run_slate = "slate" in args.libs
run_dp = "dplasma" in args.libs

# Example #1: Cholesky weak scaling with DLAF:
# Note params/weak_params entries can be a list or a single value (which is automatically converted to a list of a value).
# The following benchmark is executed for these cases (using (m_sz_1node, mb_sz) notation):
# (10240, 256)
# (10240, 512)
# (20480, 256)
# (20480, 512)
# for rpn = 1 and 2
# (5120, 64)
# (5120, 128)
# only for rpn = 2

if run_dlaf:
    run = mp.WeakScaling(system, "Cholesky_weak", "job_dlaf", nodes_arr, time0, time)
    run.add(
        mp.chol,
        "dlaf",
        libpaths["dlaf"],
        {"rpn": [1, 2], "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
    )
    run.add(
        mp.chol,
        "dlaf",
        libpaths["dlaf"],
        {"rpn": 2, "mb_sz": [64, 128]},
        {"m_sz": 5120},
        approx,
        nruns,
    )
    run.submit(run_dir, debug=debug)

# Example #2: Cholesky weak scaling with Slate:

if run_slate:
    run = mp.WeakScaling(system, "Cholesky_weak", "job_slate", nodes_arr, time0, time)
    run.add(
        mp.chol,
        "slate",
        libpaths["slate"],
        {"rpn": [1, 2], "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
    )
    run.submit(run_dir, debug=debug)

# Example #3: Trsm weak scaling with DPlasma:
# Note: n_sz = None means that n_sz = m_sz (See miniapp.trsm documentation)

if run_dp:
    run = mp.WeakScaling(system, "Trsm_weak", "job_dp", nodes_arr, time0, time)
    run.add(
        mp.trsm,
        "dplasma",
        libpaths["dplasma"],
        {"rpn": 1, "mb_sz": [256, 512], "n_sz": None},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
    )
    run.add(
        mp.trsm,
        "dplasma",
        libpaths["dplasma"],
        {"rpn": 1, "mb_sz": [256, 512]},
        {"m_sz": 20480, "n_sz": 10240},
        approx,
        nruns,
    )
    run.submit(run_dir, debug=debug)

# Example #3: GenToStd weak scaling with DLAF:

if run_dlaf:
    run = mp.WeakScaling(system, "Gen2Std_weak", "job_g2s_dlaf", nodes_arr, time0, time)
    run.add(
        mp.gen2std,
        "dlaf",
        libpaths["dlaf"],
        {"rpn": 1, "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
    )
    run.submit(run_dir, debug=debug)

# Example #4: Compare two versions:

if run_dlaf:
    run = mp.WeakScaling(system, "Cholesky_weak", "job_comp_dlaf", nodes_arr, time0, time)
    run.add(
        mp.chol,
        "dlaf",
        "<path_V1>",
        {"rpn": [1, 2], "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
        suffix="V1",
    )
    run.add(
        mp.chol,
        "dlaf",
        "<path_V2>",
        {"rpn": [1, 2], "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
        suffix="V2",
    )
    run.submit(run_dir, debug=debug)

# Example #5: Combined:

run = mp.WeakScaling(system, "Combined_weak", "job", nodes_arr, time0, time)
if run_dlaf:
    run.add(
        mp.chol,
        "dlaf",
        libpaths["dlaf"],
        {"rpn": 2, "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
    )
if run_slate:
    run.add(
        mp.chol,
        "slate",
        libpaths["slate"],
        {"rpn": 2, "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
    )
if run_dp:
    run.add(
        mp.chol,
        "dplasma",
        libpaths["dplasma"],
        {"rpn": 1, "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
    )
if run_mkl:
    run.add(
        mp.chol,
        "scalapack",
        libpaths["scalapack-mkl"],
        {"rpn": 36, "mb_sz": [64, 128]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
        suffix="mkl",
    )
if run_libsci:
    run.add(
        mp.chol,
        "scalapack",
        libpaths["scalapack-libsci"],
        {"rpn": 36, "mb_sz": [64, 128]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
        suffix="libsci",
    )
run.print()
run.submit(run_dir, debug=debug)

# Example #6: rpn fixed for the entire job as for m100 "Multiple rpn in same job" is False

run = mp.WeakScaling(systems.cineca["m100"], "Combined_weak", "job_m100", nodes_arr, time0, time)
if run_dlaf:
    run.add(
        mp.chol,
        "dlaf",
        libpaths["dlaf"],
        {"rpn": 4, "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
    )
if run_slate:
    run.add(
        mp.chol,
        "slate",
        libpaths["slate"],
        {"rpn": 4, "mb_sz": [256, 512]},
        {"m_sz": [10240, 20480]},
        approx,
        nruns,
    )
run.submit(run_dir, debug=debug)

# Example #7: Customized setup
# Note: In case more customization is needed each job can be setup manually:
from math import sqrt

run_name = "Cholesky_weak"
m_1node = 10240
mb_sz_arr = [128, 256]


def getTime(nodes):
    return time0 + int(time * sqrt(nodes))


def getSize(nodes):
    return round(m_1node * sqrt(nodes) / approx) * approx


for nodes in nodes_arr:
    if run_dlaf:
        job_text = mp.JobText(system, run_name, nodes, getTime(nodes), "job_custom_dlaf")

        for mb_sz in mb_sz_arr:
            m_sz = getSize(nodes)
            job_text.addCommand(
                mp.chol,
                lib="dlaf",
                build_dir=libpaths["dlaf"],
                rpn=2,
                m_sz=m_sz,
                mb_sz=mb_sz,
                nruns=nruns,
                suffix=f"rpn=2",
            )

            job_text.submitJobs(run_dir, debug=debug)

    if run_dp:
        job_text = mp.JobText(system, run_name, nodes, getTime(nodes), "job_custom_dp")

        for mb_sz in mb_sz_arr:
            m_sz = getSize(nodes)
            job_text.addCommand(
                mp.chol,
                lib="dplasma",
                build_dir=libpaths["dplasma"],
                rpn=1,
                m_sz=m_sz,
                mb_sz=mb_sz,
                nruns=nruns,
                suffix=f"rpn=1",
            )

        job_text.submitJobs(run_dir, debug=debug)

    if run_mkl:
        job_text = mp.JobText(system, run_name, nodes, getTime(nodes), "job_custom_mkl")

        for mb_sz in mb_sz_arr:
            m_sz = getSize(nodes)
            job_text.addCommand(
                mp.chol,
                lib="scalapack",
                build_dir=libpaths["scalapack-mkl"],
                rpn=36,
                m_sz=m_sz,
                mb_sz=mb_sz,
                nruns=nruns // 2,
                suffix="mkl_rpn=36",
            )

        job_text.submitJobs(run_dir, debug=debug)
