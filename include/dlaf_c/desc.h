//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

struct DLAF_descriptor {
  int m;
  int n;
  int mb;
  int nb;
  int isrc;
  int jsrc;
  int i;
  int j;
  int ld;
};
