#!/usr/bin/env python3

from itertools import product
import argparse
import miniapps as mp
import systems
from math import sqrt

run_name = "gen2std_weak"
system = systems.cscs["daint-gpu"]
root_dir = "/scratch/snx3000/simbergm/dlaf-202109-deliverable-benchmarks"
dlaf_build_dir = f"{root_dir}/DLA-Future/build"
dp_build_dir = f"{root_dir}/dplasma/build"
sl_build_dir = f"{root_dir}/slate/build/"
run_dir = f"{root_dir}/results/{run_name}"
time_min = 10
time_512 = 30
nruns = 10
ranks_per_node = 1
nodes_arr = [2, 8, 32, 128, 512]
nodes_arr += [1, 4, 16, 64, 256]
m_1node = 20480
m_2node = 28672
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

time_min *= len(mb_sz_arr)
time_512 *= len(mb_sz_arr)

def get_time(nodes):
    return time_min + int(time_512 * sqrt(nodes / 512))


def get_size(nodes):
    if sqrt(nodes) == int(sqrt(nodes)):
        m_sz = m_1node * int(sqrt(nodes))
    elif sqrt(nodes / 2) == int(sqrt(nodes / 2)):
        m_sz = m_2node * int(sqrt(nodes / 2))
    else:
        m_sz = None
    return m_sz


for nodes in nodes_arr:
    if run_dlaf:
        job_text = mp.init_job_text(system, run_name, nodes, get_time(nodes))

        for mb_sz in mb_sz_arr:
            m_sz = get_size(nodes)
            if m_sz is None:
                print("OUCH")
                continue

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
            )

        if args.debug:
            print(job_text)
        else:
            mp.submit_jobs(run_dir, nodes, job_text, suffix="gpu")

    if run_slate:
        job_text = mp.init_job_text(system, run_name, nodes, get_time(nodes))

        for mb_sz in mb_sz_arr:
            m_sz = get_size(nodes)
            if m_sz is None:
                print("OUCH")
                continue

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
                extra_flags="",
            )

        if args.debug:
            print(job_text)
        else:
            mp.submit_jobs(run_dir, nodes, job_text, suffix="gpu")
