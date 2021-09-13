#!/usr/bin/env python3

from itertools import product
import miniapps as mp
import systems

run_name = "cholesky_strong"
system = systems.cscs["daint-mc"]

dlaf_build_dir = "/project/csstaff/ialberto/workspace/dla-future.master/builds/daint"
dp_build_dir = "/scratch/e1000/rasolca/dplasma/build"
sl_build_dir = "/scratch/e1000/rasolca/slate-2020.10.00/build/"

run_dir = f"/scratch/snx3000/ialberto/20210901-benchmark-PRACE/{run_name}"

time_min = 400
nruns = 10
ranks_per_node = 2
nodes_arr = [1, 2, 4, 8, 16, 32]
m_sz_arr = [10240]  # , 20480, 40960]
mb_sz_arr = [256, 384, 512]

for nodes in nodes_arr:
    job_text = mp.init_job_text(system, run_name, nodes, time_min)

    for (
        m_sz,
        mb_sz,
    ) in product(m_sz_arr, mb_sz_arr):

        job_text += mp.chol(
            system,
            "dlaf",
            dlaf_build_dir,
            nodes,
            ranks_per_node,
            m_sz,
            mb_sz,
            nruns,
            suffix=f"rpn={ranks_per_node}",
        )

    # debugging
    # print(job_text)
    # break

    mp.submit_jobs(run_dir, nodes, job_text, suffix=ranks_per_node)
