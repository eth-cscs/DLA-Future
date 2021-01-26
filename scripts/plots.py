#!/usr/bin/env python3
# coding: utf-8

import postprocess as pp

df = pp.parse_jobs("/home/teonnik/projects/dlaf/mpi_mechanisms/test/mpimech_18Jan")
df_grp = pp.calc_chol_metrics(df)
pp.gen_chol_plots(df_grp)
