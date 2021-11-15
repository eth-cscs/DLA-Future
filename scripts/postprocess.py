import argparse
import os
import re
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from parse import parse


def _gen_nodes_plot(
    plt_type,
    plt_routine,
    title,
    df,
    combine_mb=False,
    filts=None,
    replaces=None,
    styles=None,
    subplot_args=None,
    fill_area=True,
):
    """
    Args:
        plt_type:       ppn | time
        plt_routine:    chol | trsm | hegst It is used to filter data.
        title:          title of the plot
        df:             the pandas.DataFrame with the data for the plot
        combine_mb:     bool indicates if different mb has to be included in the same plot
        filts:          list of regex for selecting benchmark names to plot
        replaces:       list of (regex_replace_rule, newtext) to apply to benchmark names for the legend
        styles:         list of (regex, dict()) where dict() contains kwargs valid for the plot
        subplot_args:   kwargs to pass to pyplot.subplots
        fill_area:      switch on/off the min-max area on plots
    """
    if subplot_args is None:
        subplot_args = dict()
    fig, ax = plt.subplots(**subplot_args)

    if combine_mb:
        it_space = df.groupby(["block_rows", "bench_name"])
    else:
        it_space = df.groupby(["bench_name"])

    plotted = False

    for x, lib_data in it_space:
        if combine_mb:
            mb = x[0]
            bench_name = x[1] + f"_{mb}"
        else:
            bench_name = x

        # Filter by routine
        if not bench_name.startswith(plt_routine):
            continue

        # filter series by name
        if filts != None:
            flag = False
            for filt in filts:
                if re.search(filt, bench_name):
                    flag = True
                    break
            if not flag:
                continue

        # setup style applying each config in order as they appear in the list
        # i.e. the last overwrites the first (in case of regex match)
        bench_style = dict(linestyle="-", marker=".")  # default style
        if styles != None:
            for (bench_regex, style) in styles:
                if re.search(bench_regex, bench_name):
                    bench_style |= style

        # benchmark name update happens just before plotting

        # remove routine prefix
        bench_name = bench_name[bench_name.find("_") + 1 :]

        # benchmark name replacement as specified by the user
        if replaces != None:
            for replace in replaces:
                bench_name = re.sub(replace[0], replace[1], bench_name)

        line_color = ax.plot(
            lib_data["nodes"],
            lib_data[f"{plt_type}_mean"],
            label=bench_name,
            **bench_style,
        )[0].get_color()

        if fill_area:
            ax.fill_between(
                lib_data["nodes"],
                lib_data[f"{plt_type}_min"],
                lib_data[f"{plt_type}_max"],
                alpha=0.2,
                color=line_color,
            )
        plotted = True

    if plotted:
        ax.set_title(title)

        ax.set_xlabel("nodes")
        ax.set_ylabel("GFlops/node" if plt_type == "ppn" else "Time [s]")

        nodes = df["nodes"].sort_values().unique()
        ax.set_xticks(nodes)
        ax.set_xticklabels([f"{x:d}" for x in nodes])

        ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    return plotted, fig, ax


class NodePlotWriter:
    """
    Helper generator object that creates plot with `_gen_nodes_plot`, proxies to it
    all the arguments and allow manipulation of fig and ax before saving it to a file
    with the specified filename.

    example usage:

    ```python
    with NodePlotWriter(filename, "ppn", "chol", title, df, **proxy_args) as (fig, ax):
        # log scale for ax axis
        if logx: ax.set_xscale("log", base=2)

        # alphabetical order for the legend
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, ncol=1, prop={"size": 13})
    ```

    See `_gen_nodes_plot` for details about parameters.
    """

    def __init__(self, filename, plt_type, plt_routine, title, df, **gen_plot_args):
        self.filename = filename
        self.plotted, self.fig, self.ax = _gen_nodes_plot(plt_type, plt_routine, title, df, **gen_plot_args)

    def __enter__(self):
        return (self.fig, self.ax)

    def __exit__(self, type, value, traceback):
        if self.plotted:
            self.fig.savefig(f"{self.filename}.png", dpi=300)
        plt.close(self.fig)


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
    elif bench_name.startswith("hegst_slate"):
        pstr_arr = ["input:{}hegst"]
        pstr_res = "d {} {} lower {matrix_rows:d} {:d} {block_rows:d} {grid_rows:d} {grid_cols:d} {:d} NA {time:g} NA no check"
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
    elif bench_name.startswith("chol_scalapack"):
        pstr_arr = ["PROBLEM PARAMETERS:"]
        pstr_res = "{time_ms:g}ms {perf:g}GFlop/s {matrix_rows:d} ({block_rows:d}, {block_cols:d}) ({grid_rows:d}, {grid_cols:d})"
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
            if bench_name.startswith("chol_slate"):
                rd["block_cols"] = rd["block_rows"]
                rd["matrix_cols"] = rd["matrix_rows"]
            elif bench_name.startswith("trsm_slate"):
                rd["block_cols"] = rd["block_rows"]
            elif bench_name.startswith("hegst_slate"):
                ops = pow(rd["matrix_rows"], 3)  # TODO: Check. Assuming double.
                rd["perf"] = (ops / rd["time"]) / 1e9
            elif bench_name.startswith("chol_scalapack"):
                rd["time"] = rd["time_ms"] / 1000
                rd["matrix_cols"] = rd["matrix_rows"]
            rd["perf_per_node"] = rd["perf"] / nodes

            # makes _calc_*_metrics work
            #
            # Note: DPLASMA trsm miniapp does not respect `--nruns`. This is a workaround
            # to make _calc_metrics not skipping the first run, the only one available, by
            # not setting 'run_index' field (=NaN).
            if not "dlaf" in bench_name and not bench_name.startswith("trsm_dplasma"):
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
# If distinguish_dir is True the bench name is prepended with the directory name
# This option is useful when comparing the results of different directories with the same bench_names.
def parse_jobs(data_dirs, distinguish_dir=False):
    if not isinstance(data_dirs, list):
        data_dirs = [data_dirs]
    data = []
    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(os.path.expanduser(data_dir)):
            for f in files:
                if f.endswith(".out"):
                    nodes = int(os.path.basename(subdir))
                    benchname = f[:-4]
                    if distinguish_dir:
                      benchname += "@" + data_dir

                    with open(os.path.join(subdir, f), "r") as fout:
                        data.extend(_parse_line_based(fout, benchname, nodes))

    return pd.DataFrame(data)


# Read --path command line arguments (default = ".")
# and call parse_jobs on the given directories.
# exit is called if no results are found.
def parse_jobs_cmdargs(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--path",
        action='append',
        help="Plot results from this directory.",
    )
    parser.add_argument(
        "--distinguish-dir",
        action='store_true',
        help="Add path name to bench name. Note it works better with short relative paths.",
    )
    args = parser.parse_args()
    paths = args.path
    if paths == None:
        paths = ['.']

    df = parse_jobs(paths, args.distinguish_dir)
    if df.empty:
        print('Parsed zero results, is the path correct? (path is "' + args.path + '")')
        exit(1)

    return df


def calc_chol_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "nodes", "bench_name"], df)


def calc_trsm_metrics(df):
    return _calc_metrics(
        ["matrix_rows", "matrix_cols", "block_rows", "nodes", "bench_name"], df
    )


def calc_gen2std_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "nodes", "bench_name"], df)


def gen_chol_plots(
    df,
    logx=False,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=None,
    customize_time=None,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
    """
    if combine_mb:
        it_space = df.groupby(["matrix_rows"])
    else:
        it_space = df.groupby(["matrix_rows", "block_rows"])

    for x, grp_data in it_space:
        if combine_mb:
            m = x
        else:
            m = x[0]
            mb = x[1]

        title = f"Cholesky: strong scaling ({m} x {m})"
        filename_ppn = f"chol_ppn_{m}"
        filename_time = f"chol_time_{m}"
        if not combine_mb:
            title += f", block_size = {mb} x {mb}"
            filename_ppn += f"_{mb}"
            filename_time += f"_{mb}"
        if filename_suffix != None:
            filename_ppn += f"_{filename_suffix}"
            filename_time += f"_{filename_suffix}"

        with NodePlotWriter(
            filename_ppn, "ppn", "chol", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_ppn:
                customize_ppn(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)

        with NodePlotWriter(
            filename_time, "time", "chol", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_time:
                customize_time(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)


def gen_chol_plots_weak(
    df,
    weak_rt_approx,
    logx=False,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=None,
    customize_time=None,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
    """
    df = df.assign(
        weak_rt=[
            int(round(x[0] / math.sqrt(x[1]) / weak_rt_approx)) * weak_rt_approx
            for x in zip(df["matrix_rows"], df["nodes"])
        ]
    )

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
        if filename_suffix != None:
            filename_ppn += f"_{filename_suffix}"
            filename_time += f"_{filename_suffix}"

        with NodePlotWriter(
            filename_ppn, "ppn", "chol", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_ppn:
                customize_ppn(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)

        with NodePlotWriter(
            filename_time, "time", "chol", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_time:
                customize_time(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=10)


def gen_trsm_plots(
    df,
    logx=False,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=None,
    customize_time=None,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
    """
    if combine_mb:
        it_space = df.groupby(["matrix_rows", "matrix_cols"])
    else:
        it_space = df.groupby(["matrix_rows", "matrix_cols", "block_rows"])

    for x, grp_data in it_space:
        if combine_mb:
            m, n = x
        else:
            m, n, mb = x

        title = f"TRSM: strong scaling ({m} x {n})"
        filename_ppn = f"trsm_ppn_{m}_{n}"
        filename_time = f"trsm_time_{m}_{n}"
        if not combine_mb:
            title += f", block_size = {mb} x {mb}"
            filename_ppn += f"_{mb}"
            filename_time += f"_{mb}"
        if filename_suffix != None:
            filename_ppn += f"_{filename_suffix}"
            filename_time += f"_{filename_suffix}"

        with NodePlotWriter(
            filename_ppn, "ppn", "trsm", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_ppn:
                customize_ppn(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)

        with NodePlotWriter(
            filename_time, "time", "trsm", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_time:
                customize_time(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)


def gen_trsm_plots_weak(
    df,
    weak_rt_approx,
    logx=False,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=None,
    customize_time=None,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
    """
    df = df.assign(
        weakrt_rows=[
            int(round(x[0] / math.sqrt(x[1]) / weak_rt_approx)) * weak_rt_approx
            for x in zip(df["matrix_rows"], df["nodes"])
        ],
        weakrt_cols=[
            int(round(x[0] / math.sqrt(x[1]) / weak_rt_approx)) * weak_rt_approx
            for x in zip(df["matrix_cols"], df["nodes"])
        ],
    )

    if combine_mb:
        it_space = df.groupby(["weakrt_rows", "weakrt_cols"])
    else:
        it_space = df.groupby(["weakrt_rows", "weakrt_cols", "block_rows"])

    for x, grp_data in it_space:
        if combine_mb:
            weakrt_m, weakrt_n = x
        else:
            weakrt_m, weakrt_n, mb = x

        title = f"TRSM: weak scaling ({weakrt_m} x {weakrt_n})"
        filename_ppn = f"trsm_ppn_{weakrt_m}_{weakrt_n}"
        filename_time = f"trsm_time_{weakrt_m}_{weakrt_n}"
        if not combine_mb:
            title += f", block_size = {mb} x {mb}"
            filename_ppn += f"_{mb}"
            filename_time += f"_{mb}"
        if filename_suffix != None:
            filename_ppn += f"_{filename_suffix}"
            filename_time += f"_{filename_suffix}"

        with NodePlotWriter(
            filename_ppn, "ppn", "trsm", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_ppn:
                customize_ppn(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)

        with NodePlotWriter(
            filename_time, "time", "trsm", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_time:
                customize_time(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)


def gen_gen2std_plots(
    df,
    logx=False,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=None,
    customize_time=None,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
    """
    if combine_mb:
        it_space = df.groupby(["matrix_rows"])
    else:
        it_space = df.groupby(["matrix_rows", "block_rows"])

    for x, grp_data in it_space:
        if combine_mb:
            m = x
        else:
            m = x[0]
            mb = x[1]

        title = f"HEGST: strong scaling ({m} x {m})"
        filename_ppn = f"gen2std_ppn_{m}"
        filename_time = f"gen2std_time_{m}"
        if not combine_mb:
            title += f", block_size = {mb} x {mb}"
            filename_ppn += f"_{mb}"
            filename_time += f"_{mb}"
        if filename_suffix != None:
            filename_ppn += f"_{filename_suffix}"
            filename_time += f"_{filename_suffix}"

        with NodePlotWriter(
            filename_ppn, "ppn", "hegst", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_ppn:
                customize_ppn(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)

        with NodePlotWriter(
            filename_time, "time", "hegst", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_time:
                customize_time(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)


def gen_gen2std_plots_weak(
    df,
    weak_rt_approx,
    logx=False,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=None,
    customize_time=None,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
    """
    df = df.assign(
        weak_rt=[
            int(round(x[0] / math.sqrt(x[1]) / weak_rt_approx)) * weak_rt_approx
            for x in zip(df["matrix_rows"], df["nodes"])
        ]
    )

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

        title = f"HEGST: weak scaling ({weak_rt} x {weak_rt})"
        filename_ppn = f"gen2std_ppn_{weak_rt}"
        filename_time = f"gen2std_time_{weak_rt}"
        if not combine_mb:
            title += f", block_size = {mb} x {mb}"
            filename_ppn += f"_{mb}"
            filename_time += f"_{mb}"
        if filename_suffix != None:
            filename_ppn += f"_{filename_suffix}"
            filename_time += f"_{filename_suffix}"

        with NodePlotWriter(
            filename_ppn, "ppn", "hegst", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_ppn:
                customize_ppn(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)

        with NodePlotWriter(
            filename_time, "time", "hegst", title, grp_data, combine_mb=combine_mb, **proxy_args
        ) as (fig, ax):
            if customize_time:
                customize_time(fig, ax)
            if logx:
                ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=10)
