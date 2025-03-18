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

df = pp.parse_jobs_cmdargs(description="Plot bt_band2trid strong scaling benchmarks.")

df_grp = pp.calc_bt_band2trid_metrics(df)
pp.gen_bt_band2trid_plots_strong(df_grp)
pp.gen_bt_band2trid_plots_strong(df_grp, combine_mb=True)
