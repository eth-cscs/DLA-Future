#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product

#%matplotlib

sns.set_theme()


# ## Collect data for Cholesky master vs nbmpi

# In[19]:


RE_FLOAT = "\d+(?:\.\d+)?"
regex_str = f"\[(\d+)\] ({RE_FLOAT})s ({RE_FLOAT})GFlop/s \((\d+), (\d+)\) \((\d+), (\d+)\) \((\d+), (\d+)\) \s?(\d+)"

nodes_arr = [8, 16, 32, 64]
suffixes_arr = [
    f"nbmpi_{q}_{m}_{p}"
    for q, m, p in product(
        ["shared", "default"], ["polling", "yielding"], ["mpi", "default"]
    )
]
suffixes_arr.append("master")

# Iterate over benchmark sets and node folders and parse output
data = []
for nodes, suffix in product(nodes_arr, suffixes_arr):
    bench_name = f"chol_{suffix}.txt"
    out_file = os.path.join("data", str(nodes), bench_name)
    with open(out_file, "r") as fout:
        for line in fout.readlines():
            reg = re.match(regex_str, line)
            if reg:
                raw_data = reg.groups()
                data.append(
                    {
                        "bench_name": bench_name,
                        "nodes": nodes,
                        "run_index": int(raw_data[0]),
                        "matrix_size": int(raw_data[3]),
                        "block_size": int(raw_data[5]),
                        "grid_rows": int(raw_data[7]),
                        "grid_cols": int(raw_data[8]),
                        "time": float(raw_data[1]),
                        "perf": float(raw_data[2]),
                        "perf_per_node": float(raw_data[2]) / nodes,
                    }
                )

df = pd.DataFrame(data)


# In[26]:


df_runs = (
    df.loc[df["run_index"] != 0]
    .groupby(["matrix_size", "block_size", "nodes", "bench_name"])
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

# In[ ]:


df_groups = df_runs.groupby(["matrix_size", "block_size"])
for (m, mb), grp_data in df_groups:
    fig, ax = plt.subplots()
    for bench_name, lib_data in grp_data.groupby(["bench_name"]):
        if "shared" in bench_name:
            continue
        lib_data.plot(
            ax=ax,
            x="nodes",
            y="ppn_mean",
            marker=".",
            linestyle="-",
            label=bench_name,
        )
        ax.fill_between(
            lib_data["nodes"],
            lib_data["ppn_min"],
            lib_data["ppn_max"],
            alpha=0.2,
        )

    ax.set_ylabel(f"GFlops/node\n(mb={mb})")
    ax.set_xlabel(f"nodes")
    ax.set_xticks(nodes_arr)
    ax.set_title(f"Cholesky: matrix_size = {m}, block_size = {mb}")
    ax.legend(loc="upper right",  prop={'size': 6})
    fig.savefig(f'chol_{m}_{mb}.png', dpi=300)
