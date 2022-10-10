#!/usr/bin/env python3

# This file is an example on how to use the miniapp module.
# Please do not add gen scripts used for benchmarks into the source repository,
# they should be kept with the result produced.

import argparse
import datetime
import miniapps as mp
import os
import systems

parser = argparse.ArgumentParser(description="Run strong scaling benchmarks.")
parser.add_argument(
    "--debug",
    help="Don't submit jobs, only create job scripts instead.",
    action="store_true",
)
parser.add_argument(
    "--paths",
    help="Run benchmarks at these paths.",
    nargs="+",
    default=[],
)
parser.add_argument(
    "--config",
    help="Use MC or GPU configuration.",
)
parser.add_argument(
    "--rundir",
    help="Run benchmarks in this directory.",
)
args = parser.parse_args()

if not args.paths:
    print("Need at least one path to compare, got none.")
    exit(1)

system = {}
rpn = []
mb_sz = []

if args.config == "mc":
    system = systems.cscs["daint-mc"]
    rpn = [2]
    mb_sz = [128, 256, 512]
elif args.config == "gpu":
    system = systems.cscs["daint-gpu"]
    rpn = [1]
    mb_sz = [512, 1024]
else:
    print("--config needs to be either \"mc\" or \"gpu\"")
    exit(1)

# Common parameters
m_sz = [10240, 20480]

time = 600  # minutes
nruns = 5
nodes_arr = [1, 2, 4, 8, 16, 32]
# nodes_arr = [1, 2, 4]

debug = args.debug
run_dir = args.rundir

print("Using config: " + args.config)
print("Using run directory: " + run_dir)
print("Using paths: " + str(args.paths))

common_prefix = os.path.commonprefix(args.paths)

run = mp.StrongScaling(system, "cholesky_strong", "job_dlaf", nodes_arr, time)
for path in args.paths:
    suffix = path[len(common_prefix):]
    suffix = suffix[:16]
    run.add(
        mp.chol,
        "dlaf",
        path,
        {"rpn": rpn, "m_sz": m_sz, "mb_sz": mb_sz},
        nruns,
        suffix=suffix,
    )
run.submit(os.path.join(run_dir, "cholesky_strong"), debug=debug)

run = mp.StrongScaling(system, "trsm_strong", "job_dlaf", nodes_arr, time)
for path in args.paths:
    suffix = path[len(common_prefix):]
    suffix = suffix[:16]
    run.add(
        mp.trsm,
        "dlaf",
        path,
        {"rpn": rpn, "m_sz": m_sz, "mb_sz": mb_sz, "n_sz": None},
        nruns,
        suffix=suffix,
    )
run.submit(os.path.join(run_dir, "trsm_strong"), debug=debug)

run = mp.StrongScaling(system, "gen2std_strong", "job_dlaf", nodes_arr, time)
for path in args.paths:
    suffix = path[len(common_prefix):]
    suffix = suffix[:16]
    run.add(
        mp.gen2std,
        "dlaf",
        path,
        {"rpn": rpn, "m_sz": m_sz, "mb_sz": mb_sz},
        nruns,
        suffix=suffix,
    )
run.submit(os.path.join(run_dir, "gen2std_strong"), debug=debug)
