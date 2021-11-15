#!/usr/bin/env python3
# coding: utf-8

import postprocess as pp

df = pp.parse_jobs_cmdargs(description="Plot gen2std strong scaling benchmarks.")

df_grp = pp.calc_gen2std_metrics(df)
pp.gen_gen2std_plots(df_grp)
pp.gen_gen2std_plots(df_grp, combine_mb=True)
