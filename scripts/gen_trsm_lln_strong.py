#!/usr/bin/env python3

from itertools import product
import miniapps as mp
import systems

run_name = "trsm_lln_strong"
system = systems.cscs["daint-mc"]

dlaf_build_dir = "/project/csstaff/ialberto/workspace/dla-future.master/builds/daint"
dp_build_dir = "/scratch/e1000/rasolca/dplasma/build"
sl_build_dir = "/scratch/e1000/rasolca/slate-2020.10.00/build/"

run_dir = f"/scratch/snx3000/ialberto/20210901-benchmark-PRACE/{run_name}"

time_min = 500
nruns = 5
ranks_per_node_arr = [2]
nodes_arr = [1, 2, 4, 8, 16, 32]
m_sz_arr = [10240, 20480, 40960]
mb_sz_arr = [256, 384, 512]

for nodes in nodes_arr:
    for ranks_per_node in ranks_per_node_arr:
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

        # debugging
        # print(job_text)
        # break

        mp.submit_jobs(run_dir, nodes, job_text, suffix=ranks_per_node)
        continue

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

        # debugging
        # print(job_text)
        # break

        # mp.submit_jobs(run_dir, nodes, job_text, suffix = f"sl_{ranks_per_node}")

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

    # debugging
    # print(job_text)
    # break

    # mp.submit_jobs(run_dir, nodes, job_text, suffix = "dp")
