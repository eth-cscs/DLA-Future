#!/usr/bin/env python3
# coding: utf-8

import postprocess as pp

df = pp.parse_jobs_cmdargs(description="Plot gen2std weak scaling benchmarks.")

df_grp = pp.calc_gen2std_metrics(df)
pp.gen_gen2std_plots_weak(df_grp, 1024, logx=True)
pp.gen_gen2std_plots_weak(df_grp, 1024, logx=True, combine_mb=True)
