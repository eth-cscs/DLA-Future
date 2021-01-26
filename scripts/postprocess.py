import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from parse import parse

sns.set_theme()

# plt_type : ppn | time
def _gen_nodes_plot(plt_type, title, ylabel, file_name, df):
    fig, ax = plt.subplots()
    for bench_name, lib_data in df.groupby(["bench_name"]):
        lib_data.plot(
            ax=ax,
            x="nodes",
            y=f"{plt_type}_mean",
            marker=".",
            linestyle="-",
            label=bench_name,
        )
        ax.fill_between(
            lib_data["nodes"],
            lib_data[f"{plt_type}_min"],
            lib_data[f"{plt_type}_max"],
            alpha=0.2,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("nodes")
    ax.set_xticks(df["nodes"].sort_values().unique())
    ax.legend(loc="upper right", prop={"size": 6})
    ax.set_title(title)
    fig.savefig(f"{file_name}.png", dpi=300)


# Calculate mean,max,avg perf and time
def _calc_metrics(cols, df):
    return (
        df.loc[df["run_index"] != 0]
        .groupby(cols)
        .agg(
            p_mean=("perf", "mean"),
            p_min=("perf", "min"),
            p_max=("perf", "max"),
            ppn_mean=("perf_per_node", "mean"),
            ppn_min=("perf_per_node", "min"),
            ppn_max=("perf_per_node", "max"),
            time_mean=("time", "mean"),
            time_min=("time", "min"),
            time_max=("time", "max"),
            measures=("perf", "count"),
        )
        .reset_index()
    )


# {
#     "run_index":
#     "matrix_rows":
#     "matrix_cols":
#     "block_rows":
#     "block_cols":
#     "grid_rows":
#     "grid_cols":
#     "time":
#     "perf":
#     "perf_per_node":
#     "bench_name":
#     "nodes":
# }
def _parse_line_based(fout, bench_name, nodes):
    if bench_name.startswith("dlaf"):
        pstr_arr = [
            "[{run_index:d}] {time:f}s {perf:f}GFlop/s ({matrix_rows:d}, {matrix_cols:d}) ({block_rows:d}, {block_cols:d}) ({grid_rows:d}, {grid_cols:d}) {:d}"
        ]
    elif bench_name == "chol_slate":
        pstr_arr = [
            "d host task column lower {matrix_rows:d} {:d} {block_rows:d} {grid_rows:d} {grid_cols:d} {:d} NA {time:f} {perf:f} NA NA no check"
        ]
    elif bench_name == "trsm_slate":
        pstr_arr = [
            "d host task {:d} left lower notrans nonunit {matrix_rows:d} {matrix_cols:d} {:f} {block_rows:d} {grid_rows:d} {grid_cols:d} {:d} NA {time:f} {perf:f} NA NA no check"
        ]
    elif bench_name == "chol_dplasma":
        pstr_arr = [
            "#+++++ M x N x K|NRHS : {matrix_rows:d} x {matrix_cols:d} x {:d}",
            "#+++++ MB x NB : {block_rows:d} x {block_cols:d}",
            "[****] TIME(s) {time:f} : dpotrf PxQ= {grid_rows:d} {grid_cols:d} NB= {:d} N= {:d} : {perf:f} gflops - ENQ&PROG&DEST {:f} : {:f} gflops - ENQ {:f} - DEST {:f}",
        ]
    elif bench_name == "trsm_dplasma":
        pstr_arr = [
            "#+++++ M x N x K|NRHS : {matrix_rows:d} x {matrix_cols:d} x {:d}",
            "#+++++ MB x NB : {block_rows:d} x {block_cols:d}",
            "[****] TIME(s) {time:f} : dtrsm PxQ= {grid_rows:d} {grid_cols:d} NB= {block_rows:d} N= {:d} : {perf:f} gflops",
        ]

    data = []
    rd = {}
    i_pstr = 0
    for line in fout:
        pdata = parse(pstr_arr[i_pstr], " ".join(line.split()))
        if pdata:
            rd.update(pdata.named)
            i_pstr += 1

        # if all lines are matched, add the data entry
        if i_pstr == len(pstr_arr):
            rd["bench_name"] = bench_name
            rd["nodes"] = nodes
            rd["perf_per_node"] = rd["perf"] / nodes
            if bench_name == "chol_slate":
                rd["block_cols"] = rd["block_rows"]
                rd["matrix_cols"] = rd["matrix_rows"]
            elif bench_name == "trsm_slate":
                rd["block_cols"] = rd["block_rows"]

            # makes _calc_metrics work
            if not bench_name.startswith("dlaf"):
                rd["run_index"] = 1

            data.append(rd)
            i_pstr = 0
            rd = {}

    return data


# Iterate over benchmark sets and node folders and parse output
#
# <data_dir>
# |
# ├── 16 # nodes
# │   ├── job.sh
# │   ├── <bench_name_1>.out
# │   ├── <bench_name_2>.out
# |   ...
# ├── 32
# │   ├── job.sh
# │   ├── <bench_name_1>.out
# │   ├── <bench_name_2>.out
# |   ...
#
def parse_jobs(data_dir):
    data = []
    for subdir, dirs, files in os.walk(os.path.expanduser(data_dir)):
        for f in files:
            if f.endswith(".out"):
                nodes = int(os.path.basename(subdir))
                with open(os.path.join(subdir, f), "r") as fout:
                    data.extend(_parse_line_based(fout, f[:-4], nodes))

    return pd.DataFrame(data)


def calc_chol_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "nodes", "bench_name"], df)


def calc_trsm_metrics(df):
    return _calc_metrics(
        ["matrix_rows", "matrix_cols", "block_rows", "nodes", "bench_name"], df
    )


def gen_chol_plots(df):
    for (m, mb), grp_data in df.groupby(["matrix_rows", "block_rows"]):
        title = f"Cholesky: matrix_size = {m} x {m}, block_size = {mb} x {mb}"
        _gen_nodes_plot(
            "ppn",
            title,
            "GFlops/node",
            f"chol_ppn_{m}_{mb}",
            grp_data,
        )
        _gen_nodes_plot(
            "time",
            title,
            "Time [s]",
            f"chol_time_{m}_{mb}",
            grp_data,
        )


def gen_trsm_plots(df):
    for (m, n, mb), grp_data in df.groupby(
        ["matrix_rows", "matrix_cols", "block_rows"]
    ):
        title = f"TRSM: matrix_size = {m} x {n}, block_size = {mb} x {mb}"
        _gen_nodes_plot(
            "ppn",
            title,
            "GFlops/node",
            f"trsm_ppn_{m}_{n}_{mb}",
            grp_data,
        )
        _gen_nodes_plot(
            "time",
            title,
            "Time [s]",
            f"trsm_time_{m}_{n}_{mb}",
            grp_data,
        )
