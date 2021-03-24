import os
import re
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from parse import parse

sns.set_theme()

# plt_type : ppn | time
def _gen_nodes_plot(plt_type, title, ylabel, file_name, df, logx, combine_mb=False):
    fig, ax = plt.subplots()

    if combine_mb:
        it_space = df.groupby(["block_rows", "bench_name"])
    else:
        it_space = df.groupby(["bench_name"])

    for x, lib_data in it_space:
        if combine_mb:
            mb = x[0]
            bench_name = x[1] + f"_{mb}"
        else:
            bench_name = x

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
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nodes")
    ax.set_xticks(df["nodes"].sort_values().unique())
    ax.legend(loc="upper right", prop={"size": 6})
    ax.set_title(title)
    fig.savefig(f"{file_name}.png", dpi=300)
    plt.close(fig)


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
    if "dlaf" in bench_name:
        pstr_arr = []
        pstr_res = "[{run_index:d}] {time:g}s {perf:g}GFlop/s ({matrix_rows:d}, {matrix_cols:d}) ({block_rows:d}, {block_cols:d}) ({grid_rows:d}, {grid_cols:d}) {:d}"
    elif bench_name.startswith("chol_slate"):
        pstr_arr = ["input:{}potrf"]
        pstr_res = "d {} {} column lower {matrix_rows:d} {:d} {block_rows:d} {grid_rows:d} {grid_cols:d} {:d} NA {time:g} {perf:g} NA NA no check"
    elif bench_name.startswith("trsm_slate"):
        pstr_arr = ["input:{}trsm"]
        pstr_res = "d {} {} {:d} left lower notrans nonunit {matrix_rows:d} {matrix_cols:d} {:f} {block_rows:d} {grid_rows:d} {grid_cols:d} {:d} NA {time:g} {perf:g} NA NA no check"
    elif bench_name.startswith("chol_dplasma"):
        pstr_arr = [
            "#+++++ M x N x K|NRHS : {matrix_rows:d} x {matrix_cols:d} x {:d}",
            "#+++++ MB x NB : {block_rows:d} x {block_cols:d}",
        ]
        pstr_res = "[****] TIME(s) {time:g} : dpotrf PxQ= {grid_rows:d} {grid_cols:d} NB= {:d} N= {:d} : {perf:g} gflops - ENQ&PROG&DEST {:g} : {:g} gflops - ENQ {:g} - DEST {:g}"
    elif bench_name.startswith("trsm_dplasma"):
        pstr_arr = [
            "#+++++ M x N x K|NRHS : {matrix_rows:d} x {matrix_cols:d} x {:d}",
            "#+++++ MB x NB : {block_rows:d} x {block_cols:d}",
        ]
        pstr_res = "[****] TIME(s) {time:g} : dtrsm PxQ= {grid_rows:d} {grid_cols:d} NB= {block_rows:d} N= {:d} : {perf:g} gflops"
    else:
        raise ValueError("Unknown bench_name: " + bench_name)

    data = []
    rd = {}
    # used for slate and dplasma
    run_index = 0
    for line in fout:
        for pstr in pstr_arr:
            pdata = parse(pstr, " ".join(line.split()))
            if pdata:
                rd.update(pdata.named)
                run_index = 0

        pdata = parse(pstr_res, " ".join(line.split()))
        if pdata:
            rd.update(pdata.named)
            rd["bench_name"] = bench_name
            rd["nodes"] = nodes
            rd["perf_per_node"] = rd["perf"] / nodes
            if bench_name == "chol_slate":
                rd["block_cols"] = rd["block_rows"]
                rd["matrix_cols"] = rd["matrix_rows"]
            elif bench_name == "trsm_slate":
                rd["block_cols"] = rd["block_rows"]

            # makes _calc_metrics work
            if not "dlaf" in bench_name:
                rd["run_index"] = run_index
                run_index += 1

            data.append(dict(rd))

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
def parse_jobs(data_dirs):
    if not isinstance(data_dirs, list):
        data_dirs = [data_dirs]
    data = []
    for data_dir in data_dirs:
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


def gen_chol_plots(df, logx=False):
    for (m, mb), grp_data in df.groupby(["matrix_rows", "block_rows"]):
        title = f"Cholesky: matrix_size = {m} x {m}, block_size = {mb} x {mb}"
        _gen_nodes_plot(
            "ppn",
            title,
            "GFlops/node",
            f"chol_ppn_{m}_{mb}",
            grp_data,
            logx
        )
        _gen_nodes_plot(
            "time",
            title,
            "Time [s]",
            f"chol_time_{m}_{mb}",
            grp_data,
            logx
        )
def gen_chol_plots_weak(df, weak_rt_approx, logx=False, combine_mb=False):
    df = df.assign(weak_rt=[int(round(x[0] / math.sqrt(x[1]) / weak_rt_approx)) * weak_rt_approx for x in zip(df['matrix_rows'], df['nodes'])])

    if combine_mb:
        it_space = df.groupby(["weak_rt"])
    else:
        it_space = df.groupby(["weak_rt", "block_rows"])

    for x, grp_data in it_space:
        if combine_mb:
            weak_rt = x
        else:
            weak_rt = x[0]
            mb = x[1]

        title = f"Cholesky: weak scaling ({weak_rt} x {weak_rt})"
        filename_ppn = f"chol_ppn_{weak_rt}"
        filename_time = f"chol_time_{weak_rt}"
        if not combine_mb:
            title += f", block_size = {mb} x {mb}"
            filename_ppn += f"_{mb}"
            filename_time += f"_{mb}"

        _gen_nodes_plot(
            "ppn",
            title,
            "GFlops/node",
            filename_ppn,
            grp_data,
            logx,
            combine_mb
        )
        _gen_nodes_plot(
            "time",
            title,
            "Time [s]",
            filename_time,
            grp_data,
            logx,
            combine_mb
        )

def gen_trsm_plots(df, logx=False):
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
            logx
        )
        _gen_nodes_plot(
            "time",
            title,
            "Time [s]",
            f"trsm_time_{m}_{n}_{mb}",
            grp_data,
            logx
        )
