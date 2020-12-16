import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import deque

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


# Assumptions:
#
# - at least a single white space  between various data of interest
#
def _parse_line_based(lib, fout, bench_name, nodes):

    fl = "[+-]?([0-9]*[.])?[0-9]+"

    if lib == "dlaf":
        regstr = "[{}] {} {}GFlop/s ({}, {}) ({}, {}) ({}, {}) {}".format(fl)
    elif lib == "slate_chol":
        regstr = (
            "d host task column lower {} {} {} {} {} {} NA {} {} NA NA no check".format(
                fl
            )
        )
    elif lib == "slate_trsm":
        regstr = "d host task {} left lower notrans nonunit {} {} {} {} {} {} {} NA {} {} NA NA  no check".format(
            fl
        )
    elif lin == "dplasma_chol":
        regstr ="""
        "a#+++++ P x Q : {} x {} ({}/{})",
                "#+++++ M x N x K|NRHS : {} x {} x {}",
                "#+++++ MB x NB : {} x {}",
                "[****] TIME(s) {} : dpotrf PxQ= {} {} NB= {} N= {} : {} gflops - ENQ&PROG&DEST {} : {} gflops - ENQ {} - DEST {}"
        """
        regstr = "\n".join(
            [
                "#+++++ P x Q : {} x {} ({}/{})",
                "#+++++ M x N x K|NRHS : {} x {} x {}",
                "#+++++ MB x NB : {} x {}",
                "[****] TIME(s) {} : dpotrf PxQ= {} {} NB= {} N= {} : {} gflops - ENQ&PROG&DEST {} : {} gflops - ENQ {} - DEST {}",
            ]
        ).format(fl)
    elif lin == "dplasma_trsm":
        regstr = "\n".join(
            [
                "#+++++ P x Q : {} x {} ({}/{})",
                "#+++++ M x N x K|NRHS : {} x {} x {}",
                "#+++++ MB x NB : {} x {}",
                "[****] TIME(s) {} : dtrsm PxQ= {} {} NB= {} N= {} : {} gflops",
            ]
        ).format(fl)

    multi_lines = deque([], maxlen=nlines)  # a double ended queue of fixed size
    data = []
    for line in fout:
        multi_lines.append(line)
        if len(multi_lines) != nlines:  # do nothing until the queue is filled
            continue

        reg = re.match(regstr, re.sub(" +", " ", multi_lines))
        if reg:

        # merge lines and tokenize: leave digits and dots, split on white space
        split_data = re.sub("[^\d\.]", " ", " ".join(multi_lines.split())).split()
        raw_data = [
            split_data[i] for i in ind
        ]  # pick data and leave only digits and dots
        try:
            data.append(
                {
                    "matrix_rows": int(raw_data[0]),
                    "matrix_cols": int(raw_data[1]),
                    "block_rows": int(raw_data[2]),
                    "block_cols": int(raw_data[3]),
                    "grid_rows": int(raw_data[4]),
                    "grid_cols": int(raw_data[5]),
                    "time": float(raw_data[6]),
                    "perf": float(raw_data[7]),
                    "perf_per_node": float(raw_data[7]) / nodes,
                    "bench_name": bench_name,
                    "nodes": nodes,
                }
            )
        except ValueError:
            pass

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
def parse(data_dir):
    data = []
    for subdir, dirs, files in os.walk(os.path.expanduser(data_dir)):
        for f in files:
            if f.endswith(".out"):
                nodes = int(os.path.basename(subdir))
                with open(os.path.join(subdir, f), "r") as fout:
                    data.extend(_parse_line_based("dlaf", fout, f[:-4], nodes))

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
