#!/usr/bin/env python3

from itertools import product
import miniapps as mp
import systems
from math import sqrt

run_name = "trsm_lln_weak"
system = systems.cscs["daint-mc"]

dlaf_build_dir = "/project/csstaff/ialberto/workspace/dla-future.master/builds/daint"
dp_build_dir = "/scratch/e1000/rasolca/dplasma/build"
sl_build_dir = "/scratch/e1000/rasolca/slate-2020.10.00/build/"

run_dir = f"/scratch/snx3000/ialberto/20210901-benchmark-PRACE/{run_name}"

time_min = 20
time_512 = 60

nruns = 5

ranks_per_node = 2
nodes_arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

m_1node = 20480
m_2node = 28672

mb_sz_arr = [256, 384, 512]

time_min *= len(mb_sz_arr)
time_512 *= len(mb_sz_arr)

parser = argparse.ArgumentParser(description="Run trsm LLN weak scaling benchmarks.")
parser.add_argument(
    "--debug", help="Don't submit jobs, print job scripts instead.", action="store_true"
)
parser.add_argument(
    "--libs",
    help="Run miniapps for these libraries.",
    nargs="+",
    default=["dlaf"],
)
args = parser.parse_args()

debug = args.debug
run_dlaf = "dlaf" in args.libs


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
