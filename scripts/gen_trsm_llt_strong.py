#!/usr/bin/env python3

from itertools import product
import argparse
import miniapps as mp
import systems

run_name = "trsm2"
system = systems.cscs["eiger"]
dlaf_build_dir = "/scratch/e1000/rasolca/DLA-Future/build"
dp_build_dir = "/scratch/e1000/rasolca/dplasma/build"
sl_build_dir = "/scratch/e1000/rasolca/slate-2020.10.00/build/"
run_dir = f"/scratch/e1000/rasolca/DLA-Future/scripts/{run_name}"
time_min = 500
nruns = 5
ranks_per_node_arr = [1, 2, 4, 8]
nodes_arr = [1, 2, 4, 8, 16, 32]
m_sz_arr = [20480, 40960, 81920]
mb_sz_arr = [256, 384, 512, 576, 768]

parser = argparse.ArgumentParser(description="Run trsm strong scaling benchmarks.")
parser.add_argument(
    "--debug", help="Don't submit jobs, print job scripts instead.", action="store_true"
)
parser.add_argument(
    "--libs",
    help="Run miniapps for these libraries.",
    nargs="+",
    default=["dlaf", "slate", "dplasma"],
)
args = parser.parse_args()

debug = args.debug
run_dlaf = "dlaf" in args.libs
run_slate = "slate" in args.libs
run_dp = "dplasma" in args.libs

for nodes in nodes_arr:
    for ranks_per_node in ranks_per_node_arr:
        if run_dlaf:
            job_text = mp.init_job_text(system, run_name, nodes, time_min)

            for (
                m_sz,
                mb_sz,
            ) in product(m_sz_arr, mb_sz_arr):
                for n_sz in [m_sz, m_sz // 2]:

                    job_text += mp.trsm(
                        system,
                        "dlaf",
                        dlaf_build_dir,
                        nodes,
                        ranks_per_node,
                        m_sz,
                        n_sz,
                        mb_sz,
                        nruns,
                        suffix=f"rpn={ranks_per_node}",
                    )

            mp.submit_jobs(run_dir, nodes, job_text, debug=debug, suffix=ranks_per_node)

        if run_slate:
            job_text = mp.init_job_text(system, run_name, nodes, time_min)

            for (
                m_sz,
                mb_sz,
            ) in product(m_sz_arr, mb_sz_arr):
                for n_sz in [m_sz, m_sz // 2]:

                    job_text += mp.trsm(
                        system,
                        "slate",
                        sl_build_dir,
                        nodes,
                        ranks_per_node,
                        m_sz,
                        n_sz,
                        mb_sz,
                        nruns,
                        suffix=f"rpn={ranks_per_node}",
                        env="BLIS_JC_NT=1",
                    )

            mp.submit_jobs(
                run_dir, nodes, job_text, debug=debug, suffix=f"sl_{ranks_per_node}"
            )

    if run_dp:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)
        for m_sz, mb_sz in product(m_sz_arr, mb_sz_arr):
            for n_sz in [m_sz, m_sz // 2]:

                job_text += mp.trsm(
                    system,
                    "dplasma",
                    dp_build_dir,
                    nodes,
                    1,
                    m_sz,
                    n_sz,
                    mb_sz,
                    nruns,
                    suffix="rpn=1",
                )

        mp.submit_jobs(run_dir, nodes, job_text, debug=debug, suffix="dp")
