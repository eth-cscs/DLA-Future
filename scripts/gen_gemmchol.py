#!/usr/bin/env python3

from itertools import product
import miniapps as mp
import systems

# Benchmark parameters
# https://confluence.cscs.ch/display/COMPMAT/Tall-and-skinny+matrix+multiplication+benchmark

run_name = "gemmchol" # gemmchol-gpu
system = systems.cscs["daint-mc"] # daint-gpu
dlaf_build_dir = "~/code/dlaf/build"
run_dir = f"~/runs/{run_name}"
time_min = 40
nruns = 10
ranks_per_node_arr = [2] # 1 for gpu
nodes_arr = [36, 64, 100, 144, 196, 256]
m_sz_arr = [
    4000,  # test 1
    8000,
    12000,
    11500,  # test 2
    11750,
    12000,  # test 3
    1000,
    1050,
    1100,
]
k_sz_arr = [1000000]
mb_sz_arr = [256, 512]
variations = ["na", "no_ovlp"]

for nodes in nodes_arr:
    for ranks_per_node in ranks_per_node_arr:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)
        for (
            m_sz,
            k_sz,
            mb_sz,
            suffix,
        ) in product(m_sz_arr, k_sz_arr, mb_sz_arr, variations):

            job_text += mp.gemmchol(
                system,
                dlaf_build_dir,
                nodes,
                ranks_per_node,
                m_sz,
                k_sz,
                mb_sz,
                nruns,
                suffix=suffix
            )

        print(job_text)
        break

        # mp.submit_jobs(run_dir, nodes, job_text, suffix = ranks_per_node)
