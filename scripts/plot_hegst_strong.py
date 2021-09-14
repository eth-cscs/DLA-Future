#!/usr/bin/env python3
# coding: utf-8

import postprocess as pp

run_name = "gen2std_strong"
root_dir = "/scratch/snx3000/simbergm/dlaf-202109-deliverable-benchmarks"
run_dir = f"{root_dir}/results/{run_name}"

df = pp.parse_jobs(run_dir)

df_grp = pp.calc_gen2std_metrics(df)
pp.gen_gen2std_plots(df_grp)
