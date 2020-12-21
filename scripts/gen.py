#!/usr/bin/env python3

import miniapps as mp

run_name = "extended_mpi"
build_dir = "~/build/dlaf/r2lupte/nbmpi"
run_dir = f"~/code/dlaf/scripts/{run_name}"
time_min = 60
nruns = 10
ranks_per_node = 2
nodes_arr = [8, 16, 32, 64, 96, 128, 160]
m_sz_arr = [10240, 20480, 40960]
b_sz_arr = [256, 512]

for nodes in nodes_arr:
    job_text = mp.init_job_text(run_name, build_dir, nodes, time_min)

    for queue, mech, pool in mp.product(
        ["shared", "default"], ["polling", "yielding"], ["mpi", "default"]
    ):
        job_text += mp.chol(
            build_dir,
            nodes,
            ranks_per_node,
            nruns,
            m_sz_arr,
            b_sz_arr,
            queue,
            mech,
            pool,
            f"nbmpi_{queue}_{mech}_{pool}"
        )

    job_text += mp.chol(
        "~/build/dlaf/i4otgnd/master",
        nodes,
        ranks_per_node,
        nruns,
        m_sz_arr,
        b_sz_arr,
        "default",
        "na",
        "na",
        "master"
    )

    # debugging
    #print(job_text)
    #break

    mp.submit_job(run_dir, nodes, job_text)
