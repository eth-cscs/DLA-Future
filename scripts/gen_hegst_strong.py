#!/usr/bin/env python3

from itertools import product
import argparse
import miniapps as mp
import systems

run_name = "gen2std_strong"
system = systems.cscs["daint-gpu"]
root_dir = "/scratch/snx3000/simbergm/dlaf-202109-deliverable-benchmarks"
dlaf_build_dir = f"{root_dir}/DLA-Future/build"
dp_build_dir = f"{root_dir}/dplasma/build"
sl_build_dir = f"{root_dir}/slate/build/"
run_dir = f"{root_dir}/results/{run_name}"
time_min = 600
nruns = 10
ranks_per_node = 1
nodes_arr = [1, 2, 4, 8, 16, 32]
m_sz_arr = [10240, 20480, 40960]
mb_sz_arr = [512]

parser = argparse.ArgumentParser(description="Run hegst weak scaling benchmarks.")
parser.add_argument(
    "--debug", help="Don't submit jobs, print job scripts instead.", action="store_true"
)
parser.add_argument(
    "--libs",
    help="Run miniapps for these libraries.",
    nargs="+",
    default=["dlaf", "slate"],
)
args = parser.parse_args()

debug = args.debug
run_dlaf = "dlaf" in args.libs
run_slate = "slate" in args.libs

for nodes in nodes_arr:
    if run_dlaf:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)
        for (
            m_sz,
            mb_sz,
        ) in product(m_sz_arr, mb_sz_arr):
            job_text += mp.gen2std(
                system,
                "dlaf",
                dlaf_build_dir,
                nodes,
                ranks_per_node,
                m_sz,
                mb_sz,
                nruns,
                suffix="gpu",
                extra_flags="",
            )

            if debug:
                print(job_text)
            else:
                mp.submit_jobs(run_dir, nodes, job_text, suffix="gpu")

    if run_slate:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)
        for (
            m_sz,
            mb_sz,
        ) in product(m_sz_arr, mb_sz_arr):
            job_text += mp.gen2std(
                system,
                "slate",
                sl_build_dir,
                nodes,
                ranks_per_node,
                m_sz,
                mb_sz,
                nruns,
                suffix="gpu",
                env="",
                extra_flags="",
            )

        if debug:
            print(job_text)
        else:
            mp.submit_jobs(run_dir, nodes, job_text, suffix="gpu")
