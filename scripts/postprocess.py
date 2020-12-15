import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product

#%matplotlib

sns.set_theme()

# Checks if the miniapp parameter makes sense
def _check_miniapp_param(miniapp):
    if not (miniapp == "chol" or miniapp == "trsm"):
        raise ValueError(f"Wrong value: miniapp = {miniapp}!")


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
    # [7] 5.40432s 4238.55GFlop/s (40960, 40960) (256, 256) (4, 4) 18
    re_float = "\d+(?:\.\d+)?"
    regex_str = (
        "\[(\d+)\] "
        f"({re_float})s "
        f"({re_float})GFlop/s "
        "\((\d+), (\d+)\) "
        "\((\d+), (\d+)\) "
        "\((\d+), (\d+)\) "
        "\s?(\d+)"
    )

    data = []
    for subdir, dirs, files in os.walk(os.path.expanduser(data_dir)):
        for f in files:
            if f.endswith(".out"):
                nodes = int(os.path.basename(subdir))
                with open(os.path.join(subdir, f), "r") as fout:
                    for line in fout.readlines():
                        reg = re.match(regex_str, line)
                        if reg:
                            raw_data = reg.groups()
                            data.append(
                                {
                                    "bench_name": f[:-4],  # removes .out
                                    "nodes": nodes,
                                    "run_index": int(raw_data[0]),
                                    "matrix_rows": int(raw_data[3]),
                                    "matrix_cols": int(raw_data[4]),
                                    "block_rows": int(raw_data[5]),
                                    "block_cols": int(raw_data[6]),
                                    "grid_rows": int(raw_data[7]),
                                    "grid_cols": int(raw_data[8]),
                                    "time": float(raw_data[1]),
                                    "perf": float(raw_data[2]),
                                    "perf_per_node": float(raw_data[2]) / nodes,
                                }
                            )

    return pd.DataFrame(data)


def calc_chol_metrics(df):
    return _calc_metrics(
        ["matrix_rows", "block_rows", "nodes", "bench_name"], df
    )


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
