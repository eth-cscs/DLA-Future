#!/usr/bin/env python3

from itertools import product
import miniapps as mp
import systems
from math import sqrt

run_name = "cholesky_weak"
system = systems.cscs["daint-mc"]

dlaf_build_dir = "/project/csstaff/ialberto/workspace/dla-future.master/builds/daint"
dp_build_dir = "/project/csstaff/rasolca/build_2021_Q2/dplasma/build_mc"
sl_build_dir = "/project/csstaff/rasolca/build_2021_Q2/slate/build_mc"
mkl_build_dir = "/project/csstaff/rasolca/build_2021_Q2/lu/build_mkl"
libsci_build_dir = "/project/csstaff/rasolca/build_2021_Q2/lu/build_libsci"

run_dir = f"/scratch/snx3000/ialberto/20210901-benchmark-PRACE/{run_name}"

time_min = 20
time_512 = 60
nruns = 10
ranks_per_node = 2
nodes_arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
m_1node = 20480
m_2node = 28672
mb_sz_arr = [256, 384, 512]
mb_sz_arr_scalapack = [128, 64]

run_mkl = False
run_libsci = False
run_dlaf = True
run_slate = False
run_dp = False

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
                extra_flags="--hpx:ini=hpx.max_idle_backoff_time=0",
            )

        # debugging
        # print(job_text)
        # continue

        mp.submit_jobs(run_dir, nodes, job_text, suffix=ranks_per_node)

    if run_slate:
        job_text = mp.init_job_text(system, run_name, nodes, get_time(nodes))

        for mb_sz in mb_sz_arr:
            m_sz = get_size(nodes)
            if m_sz == None:
                print("OUCH")
                continue

            job_text += mp.chol(
                system,
                "slate",
                sl_build_dir,
                nodes,
                ranks_per_node,
                m_sz,
                mb_sz,
                nruns,
                suffix=f"rpn={ranks_per_node}",
            )

        # debugging
        print(job_text)
        # break

        mp.submit_jobs(run_dir, nodes, job_text, suffix=f"sl_{ranks_per_node}")

    if run_dp:
        job_text = mp.init_job_text(system, run_name, nodes, get_time(nodes))

        for mb_sz in mb_sz_arr:
            m_sz = get_size(nodes)
            if m_sz == None:
                print("OUCH")
                continue

            job_text += mp.chol(
                system,
                "dplasma",
                dp_build_dir,
                nodes,
                1,
                m_sz,
                mb_sz,
                nruns,
                suffix="rpn=1",
            )

        # debugging
        print(job_text)
        # break

        mp.submit_jobs(run_dir, nodes, job_text, suffix=f"dp_{ranks_per_node}")

    if run_mkl:
        job_text = mp.init_job_text(system, run_name, nodes, 2 * get_time(nodes))

        for mb_sz in mb_sz_arr_scalapack:
            m_sz = get_size(nodes)
            if m_sz == None:
                print("OUCH")
                continue

            job_text += mp.chol(
                system,
                "scalapack",
                mkl_build_dir,
                nodes,
                36,
                m_sz,
                mb_sz,
                nruns // 2,
                suffix="mkl_rpn=36",
            )

        # debugging
        print(job_text)
        # break

        mp.submit_jobs(run_dir, nodes, job_text, suffix=f"mkl_{ranks_per_node}")

    if run_libsci:
        job_text = mp.init_job_text(system, run_name, nodes, 2 * get_time(nodes))

        for mb_sz in mb_sz_arr_scalapack:
            m_sz = get_size(nodes)
            if m_sz == None:
                print("OUCH")
                continue

            job_text += mp.chol(
                system,
                "scalapack",
                libsci_build_dir,
                nodes,
                36,
                m_sz,
                mb_sz,
                nruns // 2,
                suffix="libsci_rpn=36",
            )

        # debugging
        print(job_text)
        # break

        mp.submit_jobs(run_dir, nodes, job_text, suffix=f"libsci_{ranks_per_node}")
