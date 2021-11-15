#!/usr/bin/env python3

# This file is an example on how to use the miniapp module.
# Please do not add gen scripts used for benchmarks into the source repository,
# they should be kept with the result produced.

from itertools import product
import argparse
import miniapps as mp
import systems

system = systems.cscs["daint-mc"]

libpaths = {
    "dlaf": "<path_to_dlaf>",
    "dplasma": "<path_to_dplasma>",
    "slate": "<path_to_slate>",
    "scalapack-libsci": "<path_to_libsi_miniapp>",
    "scalapack-mkl": "<path_to_mkl_miniapp>",
}

run_dir = f"~/ws/runs"

time = 400 # minutes
nruns = 10
nodes_arr = [1, 2, 4, 8, 16, 32]

parser = argparse.ArgumentParser(description="Run strong scaling benchmarks.")
parser.add_argument(
    "--debug", help="Don't submit jobs, print job scripts instead.", action="store_true"
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

# Example #1: Cholesky strong scaling with DLAF:
# Note params entries can be a list or a single value (which is automatically converted to a list of a value).
# The following benchmark is executed for these cases (using (m_sz, mb_sz) notation):
# (10240, 256)
# (10240, 512)
# (20480, 256)
# (20480, 512)
# for rpn = 1 and 2
# (5120, 64)
# (5120, 128)
# only for rpn = 2

if run_dlaf:
  run = mp.StrongScaling(system, "Cholesky_strong", nodes_arr, time)
  run.add(mp.chol, "dlaf", libpaths["dlaf"], {"rpn": [1, 2], "m_sz": [10240, 20480], "mb_sz": [256, 512]}, nruns)
  run.add(mp.chol, "dlaf", libpaths["dlaf"], {"rpn": 2, "m_sz": 5120, "mb_sz": [64, 128]}, nruns)
  run.submit(run_dir, "job_dlaf", debug=debug)

# Example #2: Cholesky strong scaling with Slate:

if run_slate:
  run = mp.StrongScaling(system, "Cholesky_strong", nodes_arr, time)
  run.add(mp.chol, "slate", libpaths["slate"], {"rpn": [1, 2], "m_sz": [10240, 20480], "mb_sz": [256, 512]}, nruns)
  run.submit(run_dir, "job_slate", debug=debug)

# Example #3: Trsm strong scaling with DPlasma:
# Note: n_sz = None means that n_sz = m_sz (See miniapp.trsm documentation)

if run_dp:
  run = mp.StrongScaling(system, "Trsm_strong", nodes_arr, time)
  run.add(mp.trsm, "dplasma", libpaths["dplasma"], {"rpn": 1, "m_sz": [10240, 20480], "mb_sz": [256, 512], "n_sz": None}, nruns)
  run.submit(run_dir, "job_dp", debug=debug)

# Example #3: GenToStd strong scaling with DLAF:

if run_dlaf:
  run = mp.StrongScaling(system, "Gen2Std_strong", nodes_arr, time)
  run.add(mp.gen2std, "dlaf", libpaths["dlaf"], {"rpn": 1, "m_sz": [10240, 20480], "mb_sz": [256, 512]}, nruns)
  run.submit(run_dir, "job_g2s_dlaf", debug=debug)

# Example #4: Compare two versions:

if run_dlaf:
  run = mp.StrongScaling(system, "Cholesky_strong", nodes_arr, time)
  run.add(mp.chol, "dlaf", "<path_V1>", {"rpn": [1, 2], "m_sz": [10240, 20480], "mb_sz": [256, 512]}, nruns, suffix="V1")
  run.add(mp.chol, "dlaf", "<path_V2>", {"rpn": [1, 2], "m_sz": [10240, 20480], "mb_sz": [256, 512]}, nruns, suffix="V2")
  run.submit(run_dir, "job_comp_dlaf", debug=debug)

# Example #5: Combined:

run = mp.StrongScaling(system, "Cholesky_strong", nodes_arr, time)
if run_dlaf:
  run.add(mp.chol, "dlaf", libpaths["dlaf"], {"rpn": 2, "m_sz": [10240, 20480], "mb_sz": [256, 512]}, nruns)
if run_slate:
  run.add(mp.chol, "slate", libpaths["slate"], {"rpn": 2, "m_sz": [10240, 20480], "mb_sz": [256, 512]}, nruns)
if run_dp:
  run.add(mp.chol, "dplasma", libpaths["dplasma"], {"rpn": 1, "m_sz": [10240, 20480], "mb_sz": [256, 512]}, nruns)
if run_mkl:
  run.add(mp.chol, "scalapack", libpaths["scalapack-mkl"], {"rpn": 36, "m_sz": [10240, 20480], "mb_sz": [64, 128]}, nruns, suffix="mkl")
if run_libsci:
  run.add(mp.chol, "scalapack", libpaths["scalapack-libsci"], {"rpn": 36, "m_sz": [10240, 20480], "mb_sz": [64, 128]}, nruns, suffix="libsci")
run.print()
run.submit(run_dir, "job", debug=debug)

# Example #6: Customized setup
# Note: In case more customization is needed each job can be setup manually:
run_name = "Cholesky_strong"
m_sz_arr = [1024, 2048]
mb_sz_arr = [128, 256]

for nodes in nodes_arr:
    if run_dlaf:
        job_text = mp.init_job_text(system, run_name, nodes, time)

        for (m_sz, mb_sz) in product(m_sz_arr, mb_sz_arr):
            job_text += mp.chol(
                system,
                "dlaf",
                libpaths["dlaf"],
                nodes,
                2,
                m_sz,
                mb_sz,
                nruns,
                suffix=f"rpn=2",
            )

            mp.submit_jobs(run_dir, nodes, job_text, debug=debug, bs_name=f"job_custom_dlaf")

    if run_dp:
        job_text = mp.init_job_text(system, run_name, nodes, time)

        for (m_sz, mb_sz) in product(m_sz_arr, mb_sz_arr):
            job_text += mp.chol(
                system,
                "dplasma",
                libpaths["dplasma"],
                nodes,
                1,
                m_sz,
                mb_sz,
                nruns,
                suffix=f"rpn=1",
            )

        mp.submit_jobs(
            run_dir, nodes, job_text, debug=debug, bs_name=f"job_custom_dp"
        )

    if run_mkl:
        job_text = mp.init_job_text(system, run_name, nodes, time)

        for (m_sz, mb_sz) in product(m_sz_arr, mb_sz_arr):
            job_text += mp.chol(
                system,
                "scalapack",
                libpaths["scalapack-mkl"],
                nodes,
                36,
                m_sz,
                mb_sz,
                nruns // 2,
                suffix="mkl_rpn=36",
            )

        mp.submit_jobs(
            run_dir, nodes, job_text, debug=debug, bs_name=f"job_custom_mkl"
        )
