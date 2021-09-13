#!/usr/bin/env python3

from itertools import product
import miniapps as mp
import systems

system = systems.cscs["daint-mc"]
run_name = "extended_mpi"
build_dir = "~/build/dlaf/r2lupte/nbmpi"
slate_build_dir = "~/build/slate"
dplasma_build_dir = "~/build/dplasma"
run_dir = f"~/code/dlaf/scripts/{run_name}"
time_min = 60
nruns = 10
ranks_per_node = 2
nodes_arr = [8, 16, 32, 64]
m_sz_arr = [10240, 20480]
n_sz_arr = [x // 2 for x in m_sz_arr]
mb_sz_arr = [256, 512]

for nodes in nodes_arr:
    job_text = mp.init_job_text(system, run_name, nodes, time_min)

    for m_sz, mb_sz, queue, mech, pool in product(
        m_sz_arr, mb_sz_arr, ["0", "1"], ["0", "1"], ["0", "1"]
    ):
        extra_flags = "{} {} {}".format(
            "--hpx:queuing=shared-priority" if queue == "1" else "",
            "--polling" if mech == "1" else "",
            "--mpipool --hpx:ini=hpx.max_idle_backoff_time=0" if pool == "1" else "",
        )
        suffix = f"nb_{queue}{mech}{pool}"
        job_text += mp.chol(
            system,
            "dlaf",
            build_dir,
            nodes,
            ranks_per_node,
            m_sz,
            mb_sz,
            nruns,
            suffix,
            extra_flags,
        )

    for m_sz, mb_sz in product(m_sz_arr, mb_sz_arr):
        job_text += mp.chol(
            system,
            "slate",
            slate_build_dir,
            nodes,
            ranks_per_node,
            m_sz,
            mb_sz,
            nruns,
        )

    for m_sz, mb_sz in product(m_sz_arr, mb_sz_arr):
        job_text += mp.chol(
            system,
            "dplasma",
            dplasma_build_dir,
            nodes,
            ranks_per_node,
            m_sz,
            mb_sz,
            nruns,
        )

    # debugging
    print(job_text)
    break

    # mp.submit_jobs(run_dir, nodes, job_text)
