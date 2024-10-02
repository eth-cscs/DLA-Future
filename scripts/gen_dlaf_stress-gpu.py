#!/usr/bin/env python3

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# This file is an example on how to use the miniapp module.
# Please do not add gen scripts used for benchmarks into the source repository,
# they should be kept with the result produced.

import argparse
import miniapps as mp
import random
import systems

system = systems.cscs["todi"]

dlafpath = "/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/dla-future-git.202410-develop_0.7.0-virbeq6o47zeihzraiblbc5zfxzafrpw/bin"
run_dir = ""

time = 60  # minutes
nruns = 20
njobs = 100
nodes_arr = [1, 2, 4, 8, 16, 32]

rpn = 4
m_szs_d = [10240, 20480, 30097, 40960]
mb_szs_d = [1024]
m_szs_z = [10240, 20480]
mb_szs_z = [1024]

extra_flags = "--pika:threads=64 --nwarmups=0 --check=last"

parser = argparse.ArgumentParser(description="Run strong scaling benchmarks.")
parser.add_argument(
    "--debug",
    help="Don't submit jobs, only create job scripts instead.",
    action="store_true",
)
args = parser.parse_args()

debug = args.debug

def createAndSubmitRun(run_dir, nodes_arr, **kwargs):
    dtype = random.choice(["d", "z"])
    if dtype == "d":
        m_szs = random.choice(m_szs_d)
        mb_szs = random.choice(mb_szs_d)
        run_dir += "/d"
    elif dtype == "z":
        m_szs = random.choice(m_szs_z)
        mb_szs = random.choice(mb_szs_z)
        run_dir += "/z"
    else:
        raise RuntimeError(f"Invalid type specified {dtype}")

    full_kwargs = kwargs.copy()
    full_kwargs["lib"] = "dlaf"
    full_kwargs["miniapp_dir"] = dlafpath
    full_kwargs["nruns"] = nruns
    full_kwargs["dtype"] = dtype

    run = mp.StrongScaling(system, "DLAF_test_strong", "job_dlaf", nodes_arr, time)

    run.add(
        mp.evp,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "min_band": None},
        **full_kwargs,
    )
    run.add(
        mp.gevp,
        params={"rpn": rpn, "m_sz": m_szs, "mb_sz": mb_szs, "min_band": None},
        **full_kwargs,
    )

    run.submit(run_dir, debug=debug)


for i in range(njobs):
    createAndSubmitRun(run_dir + "/" + str(i), [random.choice(nodes_arr)], extra_flags=extra_flags)
