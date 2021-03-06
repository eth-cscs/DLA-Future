#!/usr/bin/env python3

from itertools import product
import miniapps as mp
import systems

run_name = "cholesky2"
system = systems.cscs["eiger"]
dlaf_build_dir = "/scratch/e1000/rasolca/DLA-Future/build"
dp_build_dir = "/scratch/e1000/rasolca/dplasma/build"
sl_build_dir = "/scratch/e1000/rasolca/slate-2020.10.00/build/"
run_dir = f"/scratch/e1000/rasolca/DLA-Future/scripts/{run_name}"
time_min = 400
nruns = 5
ranks_per_node_arr = [1, 2, 4, 8]
nodes_arr = [1, 2, 4, 8, 16, 32]
m_sz_arr = [20480, 40960, 81920]
mb_sz_arr = [256, 384, 512, 576, 768]

for nodes in nodes_arr:
    for ranks_per_node in ranks_per_node_arr:
        job_text = mp.init_job_text(system, run_name, nodes, time_min)

        for m_sz, mb_sz, in product(m_sz_arr, mb_sz_arr):

            job_text += mp.chol(
                system,
                "dlaf",
                dlaf_build_dir,
                nodes,
                ranks_per_node,
                m_sz,
                mb_sz,
                nruns,
                suffix = f"rpn={ranks_per_node}"
            )

        # debugging
        # print(job_text)
        # break

        # mp.submit_jobs(run_dir, nodes, job_text, suffix = ranks_per_node)

        job_text = mp.init_job_text(system, run_name, nodes, time_min)

        for m_sz, mb_sz, in product(m_sz_arr, mb_sz_arr):

            job_text += mp.chol(
                system,
                "slate",
                sl_build_dir,
                nodes,
                ranks_per_node,
                m_sz,
                mb_sz,
                nruns,
                suffix = f"rpn={ranks_per_node}",
                env = "BLIS_JC_NT=1"
            )

        # debugging
        # print(job_text)
        # break

        # mp.submit_jobs(run_dir, nodes, job_text, suffix = f"sl_{ranks_per_node}")

    job_text = mp.init_job_text(system, run_name, nodes, time_min)
    for m_sz, mb_sz in product(m_sz_arr, mb_sz_arr):

        job_text += mp.chol(
            system,
            "dplasma",
            dp_build_dir,
            nodes,
            1,
            m_sz,
            mb_sz,
            nruns,
            suffix = "rpn=1"
        )

    # debugging
    # print(job_text)
    # break

    # mp.submit_jobs(run_dir, nodes, job_text, suffix = "dp")
