#!/usr/bin/env python3
# coding: utf-8

import postprocess as pp

df = pp.parse_jobs("~/downloads/exp_data")
df_grp = pp.calc_chol_metrics(df)
pp.gen_chol_plots(df_grp)
