#!/usr/bin/env python3

from itertools import product
import miniapps as mp

run_name = "extended_mpi"
build_dir = "~/build/dlaf/r2lupte/nbmpi"
run_dir = f"~/code/dlaf/scripts/{run_name}"
time_min = 60
nruns = 10
ranks_per_node = 2
nodes_arr = [8, 16, 32, 64]
m_sz_arr = [10240, 20480]
mb_sz_arr = [256, 512]

for nodes in nodes_arr:
    job_text = mp.init_job_text(run_name, nodes, time_min)

    for m_sz, mb_sz, queue, mech, pool in product(
        m_sz_arr, mb_sz_arr, ["0", "1"], ["0", "1"], ["0", "1"]
    ):
        job_text += mp.chol(
            f"dlaf_nb_{queue}{mech}{pool}",
            build_dir,
            nodes,
            ranks_per_node,
            m_sz,
            mb_sz,
            nruns,
        )

    n_sz_arr = [x // 2 for x in m_sz_arr]
    for m_sz, n_sz, mb_sz in product(m_sz_arr, n_sz_arr, mb_sz_arr):
        job_text += mp.trsm(
            "slate",
            build_dir,
            nodes,
            ranks_per_node,
            m_sz,
            n_sz,
            mb_sz,
            nruns,
        )

    # debugging
    print(job_text)
    break

    #mp.submit_job(run_dir, nodes, job_text)
