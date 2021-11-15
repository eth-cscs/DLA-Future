#!/usr/bin/env python3
# coding: utf-8

import postprocess as pp

df = pp.parse_jobs_cmdargs(description="Plot trsm strong scaling benchmarks.")

df_grp = pp.calc_trsm_metrics(df)
pp.gen_trsm_plots(df_grp)
pp.gen_trsm_plots(df_grp, combine_mb=True)
