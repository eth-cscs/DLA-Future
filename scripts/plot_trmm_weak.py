#!/usr/bin/env python3
# coding: utf-8

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

import postprocess as pp

df = pp.parse_jobs_cmdargs(description="Plot trmm weak scaling benchmarks.")

df_grp = pp.calc_trmm_metrics(df)
pp.gen_trmm_plots_weak(df_grp, 1024, logx=True)
pp.gen_trmm_plots_weak(df_grp, 1024, logx=True, combine_mb=True)
