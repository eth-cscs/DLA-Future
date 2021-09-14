#!/usr/bin/env python3
# coding: utf-8

import argparse
import postprocess as pp

parser = argparse.ArgumentParser(description="Plot cholesky strong scaling benchmarks.")
parser.add_argument(
    "--path",
    help="Plot results from this directory.",
    default=".",
)
args = parser.parse_args()

df = pp.parse_jobs(args.path)
if df.empty:
    print('Parsed zero results, is the path correct? (path is "' + args.path + '")')
    exit(1)

df_grp = pp.calc_chol_metrics(df)
pp.gen_chol_plots(df_grp)
