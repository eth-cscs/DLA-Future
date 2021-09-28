#!/usr/bin/env python3

from itertools import product
import argparse
import miniapps as mp
import systems

run_name = "cholesky_strong"
system = systems.cscs["daint-mc"]

libpaths = {
    "dlaf": "/users/ialberto/workspace/dla-future.master/builds/daint",
    "dplasma": "/project/csstaff/rasolca/build_2021_Q2/dplasma/build_mc",
    "slate": "/project/csstaff/rasolca/build_2021_Q2/slate/build_mc",
    "scalapack-libsci": "/project/csstaff/rasolca/build_2021_Q2/lu/build_libsci",
    "scalapack-mkl": "/project/csstaff/rasolca/build_2021_Q2/lu/build_mkl",
}

run_dir = f"/scratch/snx3000/ialberto/20210901-benchmark-PRACE/{run_name}"

time_min = 400
nruns = 10
ranks_per_node = 2
nodes_arr = [1, 2, 4, 8, 16, 32]
m_sz_arr = [10240, 20480, 40960]
mb_sz_arr = [256, 384, 512]

parser = argparse.ArgumentParser(description="Run cholesky strong scaling benchmarks.")
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

for nodes in nodes_arr:
    if run_dlaf:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)

        for (m_sz, mb_sz) in product(m_sz_arr, mb_sz_arr):
            job_text += mp.chol(
                system,
                "dlaf",
                libpaths["dlaf"],
                nodes,
                ranks_per_node,
                m_sz,
                mb_sz,
                nruns,
                suffix=f"rpn={ranks_per_node}",
            )

            mp.submit_jobs(run_dir, nodes, job_text, debug=debug, suffix=ranks_per_node)

    if run_slate:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)

        for (m_sz, mb_sz) in product(m_sz_arr, mb_sz_arr):
            job_text += mp.chol(
                system,
                "slate",
                libpaths["slate"],
                nodes,
                ranks_per_node,
                m_sz,
                mb_sz,
                nruns,
                suffix=f"rpn={ranks_per_node}",
            )

        mp.submit_jobs(
            run_dir, nodes, job_text, debug=debug, suffix=f"sl_{ranks_per_node}"
        )

    if run_dp:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)

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
                suffix=f"rpn={ranks_per_node}",
            )

        mp.submit_jobs(
            run_dir, nodes, job_text, debug=debug, suffix=f"dp_{ranks_per_node}"
        )

    if run_mkl:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)

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
            run_dir, nodes, job_text, debug=debug, suffix=f"mkl_{ranks_per_node}"
        )

    if run_libsci:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)

        for (m_sz, mb_sz) in product(m_sz_arr, mb_sz_arr):
            job_text += mp.chol(
                system,
                "scalapack",
                libpaths["scalapack-libsci"],
                nodes,
                36,
                m_sz,
                mb_sz,
                nruns // 2,
                suffix="libsci_rpn=36",
            )

        mp.submit_jobs(
            run_dir, nodes, job_text, debug=debug, suffix=f"libsci_{ranks_per_node}"
        )
