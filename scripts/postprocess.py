#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

import argparse
import os
import re
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from parse import parse, with_pattern
from pathlib import Path

default_logx = False
default_logy = False
miny0 = False
outpath = Path(".")


def _str_nnodes(x):
    if isinstance(x, float):
        if x.is_integer():
            return f"{x:.0f}"
        return f"{x:.3f}"
    return f"{x:d}"


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
        plt_routine:    chol | hegst | red2band | band2trid | trid_evp | bt_band2trid | bt_red2band | trsm | trmm | evp | gevp It is used to filter data.
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
        it_space = df.groupby("bench_name")

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
            for bench_regex, style in styles:
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
                lib_data["nodes"].values,
                lib_data[f"{plt_type}_min"].values,
                lib_data[f"{plt_type}_max"].values,
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
        ax.set_xticklabels([_str_nnodes(x) for x in nodes])

        ax.grid(axis="y", linewidth=0.5, alpha=0.5)

        if miny0:
            ax.set_ylim(0, ax.get_ylim()[1])

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
        self.filename = outpath / filename
        self.plotted, self.fig, self.ax = _gen_nodes_plot(
            plt_type, plt_routine, title, df, **gen_plot_args
        )

    def __enter__(self):
        return (self.fig, self.ax)

    def __exit__(self, type, value, traceback):
        if self.plotted:
            self.fig.savefig(f"{self.filename}.png", dpi=300)
        plt.close(self.fig)


# Calculate mean,max,avg perf and time
def _calc_metrics(cols, df):
    perf_agg_functions = dict(
        p_mean=("perf", "mean"),
        p_min=("perf", "min"),
        p_max=("perf", "max"),
        ppn_mean=("perf_per_node", "mean"),
        ppn_min=("perf_per_node", "min"),
        ppn_max=("perf_per_node", "max"),
    )
    return (
        df.loc[df["run_index"] != 0]
        .groupby(cols)
        .agg(
            **(perf_agg_functions if "perf" in df.columns else {}),
            time_mean=("time", "mean"),
            time_min=("time", "min"),
            time_max=("time", "max"),
            measures=("time", "count"),
        )
        .reset_index()
    )


@with_pattern(r"(|\s+\S+)")
def _parse_optional_text(text):
    text = text.strip()
    # TODO: Prefer empty string or None?
    if text:
        return text
    else:
        return None


additional_parsers = dict(optional_text=_parse_optional_text)
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
        # Note that the optional fields must not have a space in front of them.
        # Otherwise the space is required and parsing the optional field will
        # fail.
        alg_name = bench_name[0 : bench_name.find("_dlaf")]

        if alg_name in ["chol", "hegst", "trsm", "trmm"]:
            pstr_res = "[{run_index:d}] {time:g}s {perf:g}GFlop/s{matrix_type:optional_text} ({matrix_rows:d}, {matrix_cols:d}) ({block_rows:d}, {block_cols:d}) ({grid_rows:d}, {grid_cols:d}) {:d}{backend:optional_text}"
        if alg_name in ["red2band", "band2trid", "bt_band2trid", "bt_red2band"]:
            pstr_res = "[{run_index:d}] {time:g}s {perf:g}GFlop/s{matrix_type:optional_text} ({matrix_rows:d}, {matrix_cols:d}) ({block_rows:d}, {block_cols:d}) {band:d} ({grid_rows:d}, {grid_cols:d}) {:d}{backend:optional_text}"
        if alg_name in ["trid_evp"]:
            pstr_res = "[{run_index:d}] {time:g}s{matrix_type:optional_text} ({matrix_rows:d}, {matrix_cols:d}) ({block_rows:d}, {block_cols:d}) ({grid_rows:d}, {grid_cols:d}) {:d}{backend:optional_text}"
        if alg_name in ["evp", "gevp"]:
            pstr_res = "[{run_index:d}] {time:g}s{matrix_type:optional_text} ({matrix_rows:d}, {matrix_cols:d}) ({block_rows:d}, {block_cols:d}) {band:d} ({grid_rows:d}, {grid_cols:d}) {:d}{backend:optional_text}"
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
    elif bench_name.startswith("evp_scalapack"):
        pstr_arr = []
        pstr_res = "[{run_index:d}] Scalapack {time:g}s {matrix_type} ({matrix_rows:d}, {matrix_cols:d}) ({block_rows:d}, {block_cols:d}) ({grid_rows:d}, {grid_cols:d})"
    elif bench_name.startswith("evp_elpa"):
        stages = bench_name[8]
        pstr_arr = []
        pstr_res = (
            "[{run_index:d}] Elpa"
            + stages
            + " {time:g}s {matrix_type} ({matrix_rows:d}, {matrix_cols:d}) ({block_rows:d}, {block_cols:d}) ({grid_rows:d}, {grid_cols:d})"
        )
    else:
        raise ValueError("Unknown bench_name: " + bench_name)

    data = []
    rd = {}
    # used for slate and dplasma
    run_index = 0
    for line in fout:
        for pstr in pstr_arr:
            pdata = parse(pstr, " ".join(line.split()), additional_parsers)
            if pdata:
                rd.update(pdata.named)
                run_index = 0

        pdata = parse(pstr_res, " ".join(line.split()), additional_parsers)
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

            if "perf" in rd:
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
                    nodes = float(os.path.basename(subdir))
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
    global miny0
    global default_logx
    global default_logy
    global outpath

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--path",
        action="append",
        help="Plot results from this directory. You can pass this option several times.",
    )
    parser.add_argument(
        "--distinguish-dir",
        action="store_true",
        help="Add path name to bench name. Note it works better with short relative paths.",
    )
    parser.add_argument(
        "--miny0",
        action="store_true",
        help="Set min y limit to 0.",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        help="Set logarithmic scale for x-axis.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Set logarithmic scale for y-axis.",
    )
    parser.add_argument(
        "--out-path",
        default=".",
        help='Path to save the plots (default ".").',
    )
    args = parser.parse_args()
    paths = args.path
    miny0 = args.miny0
    default_logx = args.logx
    default_logy = args.logy
    outpath = Path(args.out_path)

    os.makedirs(outpath, exist_ok=True)

    if paths == None:
        paths = ["."]

    df = parse_jobs(paths, args.distinguish_dir)
    if df.empty:
        print("Parsed zero results, is the path correct? (paths are " + str(paths) + ")")
        exit(1)

    return df


def calc_chol_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "nodes", "bench_name"], df)


def calc_trsm_metrics(df):
    return _calc_metrics(["matrix_rows", "matrix_cols", "block_rows", "nodes", "bench_name"], df)


def calc_trmm_metrics(df):
    return _calc_metrics(["matrix_rows", "matrix_cols", "block_rows", "nodes", "bench_name"], df)


def calc_gen2std_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "nodes", "bench_name"], df)


def calc_red2band_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "band", "nodes", "bench_name"], df)


def calc_band2trid_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "band", "nodes", "bench_name"], df)


def calc_trid_evp_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "nodes", "bench_name"], df)


def calc_bt_band2trid_metrics(df):
    return _calc_metrics(["matrix_rows", "matrix_cols", "block_rows", "band", "nodes", "bench_name"], df)


def calc_bt_red2band_metrics(df):
    return _calc_metrics(["matrix_rows", "matrix_cols", "block_rows", "band", "nodes", "bench_name"], df)


def calc_evp_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "band", "nodes", "bench_name"], df)


def calc_gevp_metrics(df):
    return _calc_metrics(["matrix_rows", "block_rows", "band", "nodes", "bench_name"], df)


# Customization that add a simple legend
def add_basic_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return

    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, ncol=1, prop={"size": 10})


def _gen_plot(
    scaling,
    name,
    routine,
    filename,
    size_type,
    df,
    logx,
    logy,
    combine_mb,
    filename_suffix,
    ppn_plot=False,
    customize_ppn=None,
    time_plot=False,
    customize_time=None,
    weak_rt_approx=None,
    has_band=False,
    **proxy_args,
):
    """
    Args:
        scaling         strong | weak
        name:           name of the routine to be included in the title
        routine:        chol | hegst | red2band | band2trid | trid_evp | bt_band2trid | trsm | trmm | evp | gevp
        combine_mb:     bool indicates if different mb has to be included in the same plot
        size_type:      m | mn It indicates which sizes are relevant.
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        has_band:       switch on/off band parameter in non combined plots
    """

    if logx == None:
        logx = default_logx
    if logy == None:
        logy = default_logy

    if scaling == "weak":
        if weak_rt_approx == None:
            raise ValueError(f"Invalid weakrt_approx: {weakrt_approx}")

        if size_type == "m":
            df = df.assign(
                weakrt_rows=[
                    int(round(x[0] / math.sqrt(x[1]) / weak_rt_approx)) * weak_rt_approx
                    for x in zip(df["matrix_rows"], df["nodes"])
                ],
            )
            group_list = ["weakrt_rows"]
        elif size_type == "mn":
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
            group_list = ["weakrt_rows", "weakrt_cols"]
        else:
            raise ValueError(f"Unknown size_type: {size_type}")

    elif scaling == "strong":
        if size_type == "m":
            group_list = ["matrix_rows"]
        elif size_type == "mn":
            group_list = ["matrix_rows", "matrix_cols"]
        else:
            raise ValueError(f"Unknown size_type: {size_type}")
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    if not combine_mb:
        group_list += ["block_rows"]
        if has_band:
            group_list += ["band"]

    # silence a warning in pandas.
    if len(group_list) == 1:
        group_list = group_list[0]

    it_space = df.groupby(group_list)

    for x, grp_data in it_space:
        if size_type == "m" and combine_mb:
            # single element has to be treated differently
            m = x
        else:
            if size_type == "m":
                m = x[0]
                i = 1
            elif size_type == "mn":
                m, n = x[0:2]
                i = 2
            if not combine_mb:
                mb = x[i]
                i += 1
                if has_band:
                    # pandas use floating point if the first entry has no band.
                    b = int(x[i])
                    i += 1

        title = f"{name}: {scaling} scaling"
        filename_ppn = f"{filename}_{scaling}_ppn"
        filename_time = f"{filename}_{scaling}_time"
        if size_type == "m":
            title += f" ({m} x {m})"
            filename_ppn += f"_{m}"
            filename_time += f"_{m}"
        elif size_type == "mn":
            title += f" ({m} x {n})"
            filename_ppn += f"_{m}_{n}"
            filename_time += f"_{m}_{n}"
        if not combine_mb:
            title += f", block_size = {mb} x {mb}"
            filename_ppn += f"_{mb}"
            filename_time += f"_{mb}"
            if has_band:
                title += f", band size = {b}"
                filename_ppn += f"_{b}"
                filename_time += f"_{b}"
        if filename_suffix != None:
            filename_ppn += f"_{filename_suffix}"
            filename_time += f"_{filename_suffix}"

        if ppn_plot:
            with NodePlotWriter(
                filename_ppn,
                "ppn",
                routine,
                title,
                grp_data,
                combine_mb=combine_mb,
                **proxy_args,
            ) as (fig, ax):
                if customize_ppn:
                    customize_ppn(fig, ax)
                if logx:
                    ax.set_xscale("log", base=2)
        if time_plot:
            with NodePlotWriter(
                filename_time,
                "time",
                routine,
                title,
                grp_data,
                combine_mb=combine_mb,
                **proxy_args,
            ) as (fig, ax):
                if customize_time:
                    customize_time(fig, ax)
                if logx:
                    ax.set_xscale("log", base=2)
                if logy:
                    ax.set_yscale("log", base=10)


def gen_chol_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="Cholesky",
        routine="chol",
        filename="chol",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        **proxy_args,
    )


def gen_chol_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="Cholesky",
        routine="chol",
        filename="chol",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        **proxy_args,
    )


def gen_trsm_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="TRSM",
        routine="trsm",
        filename="trsm",
        size_type="mn",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        **proxy_args,
    )


def gen_trsm_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="TRSM",
        routine="trsm",
        filename="trsm",
        size_type="mn",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        **proxy_args,
    )


def gen_trmm_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="TRMM",
        routine="trmm",
        filename="trmm",
        size_type="mn",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        **proxy_args,
    )


def gen_trmm_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="TRMM",
        routine="trmm",
        filename="trmm",
        size_type="mn",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        **proxy_args,
    )


def gen_gen2std_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="HEGST",
        routine="hegst",
        filename="gen2std",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        **proxy_args,
    )


def gen_gen2std_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="HEGST",
        routine="hegst",
        filename="gen2std",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        **proxy_args,
    )


def gen_red2band_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="RED2B",
        routine="red2band",
        filename="red2band",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        has_band=True,
        **proxy_args,
    )


def gen_red2band_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="RED2B",
        routine="red2band",
        filename="red2band",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        has_band=True,
        **proxy_args,
    )


def gen_band2trid_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="B2T",
        routine="band2trid",
        filename="band2trid",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        has_band=True,
        **proxy_args,
    )


def gen_band2trid_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="B2T",
        routine="band2trid",
        filename="band2trid",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        has_band=True,
        **proxy_args,
    )


def gen_trid_evp_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="TridiagSolver",
        routine="trid_evp",
        filename="trid_evp",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=False,
        time_plot=True,
        customize_time=customize_time,
        **proxy_args,
    )


def gen_trid_evp_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="TridiagSolver",
        routine="trid_evp",
        filename="trid_evp",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=False,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        **proxy_args,
    )


def gen_bt_band2trid_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="BT_B2T",
        routine="bt_band2trid",
        filename="bt_band2trid",
        size_type="mn",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        has_band=True,
        **proxy_args,
    )


def gen_bt_band2trid_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="BT_B2T",
        routine="bt_band2trid",
        filename="bt_band2trid",
        size_type="mn",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        has_band=True,
        **proxy_args,
    )


def gen_bt_red2band_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="BT_RED2B",
        routine="bt_red2band",
        filename="bt_red2band",
        size_type="mn",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        has_band=True,
        **proxy_args,
    )


def gen_bt_red2band_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="BT_RED2B",
        routine="bt_red2band",
        filename="bt_red2band",
        size_type="mn",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=True,
        customize_ppn=customize_ppn,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        has_band=True,
        **proxy_args,
    )


def gen_evp_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="EVP",
        routine="evp",
        filename="evp",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=False,
        time_plot=True,
        customize_time=customize_time,
        has_band=True,
        **proxy_args,
    )


def gen_evp_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="EVP",
        routine="evp",
        filename="evp",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=False,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        has_band=True,
        **proxy_args,
    )


def gen_gevp_plots_strong(
    df,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="strong",
        name="GEVP",
        routine="gevp",
        filename="gevp",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=False,
        time_plot=True,
        customize_time=customize_time,
        has_band=True,
        **proxy_args,
    )


def gen_gevp_plots_weak(
    df,
    weak_rt_approx,
    logx=None,
    logy=None,
    combine_mb=False,
    filename_suffix=None,
    customize_ppn=add_basic_legend,
    customize_time=add_basic_legend,
    **proxy_args,
):
    """
    Args:
        customize_ppn:  function accepting the two arguments fig and ax for ppn plot customization
        customize_time: function accepting the two arguments fig and ax for time plot customization
        Default customization (ppn and time): add_basic_legend. They can be set to "None" to remove the legend.
    """
    _gen_plot(
        scaling="weak",
        name="GEVP",
        routine="gevp",
        filename="gevp",
        size_type="m",
        df=df,
        logx=logx,
        logy=logy,
        combine_mb=combine_mb,
        filename_suffix=filename_suffix,
        ppn_plot=False,
        time_plot=True,
        customize_time=customize_time,
        weak_rt_approx=weak_rt_approx,
        has_band=True,
        **proxy_args,
    )
