#!/usr/bin/env python3

import miniapps as mp
import systems

system=systems.cscs["daint-mc"]
run_name = "apex_gemmchol"
run_dir = f"~/runs/{run_name}"

# APEX job
job_text = mp.init_job_text(
    system, run_name, nodes=36, time_min=10
) + mp.gemmchol(
    system,
    build_dir="~/code/dlaf/build",
    nodes=36,
    rpn=2,
    m_sz=10000,
    k_sz=1000000,
    mb_sz=512,
    nruns=5,
    suffix="apex",
)
print(job_text)
#mp.submit_jobs(run_dir, nodes, job_text)
