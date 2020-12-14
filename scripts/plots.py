#!/usr/bin/env python3
# coding: utf-8


data_dirs = ["data"]
nodes_arr = [8, 16, 32, 64]
bench_name_arr = [
    f"chol_nbmpi_{q}_{m}_{p}"
    for q, m, p in product(
        ["shared", "default"], ["polling", "yielding"], ["mpi", "default"]
    )
]
bench_name_arr.append("chol_master")

# TODO: call parse func
