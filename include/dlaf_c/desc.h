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

/// DLA-Future descriptor
struct DLAF_descriptor {
  int m;     ///< Number of rows in the global matrix
  int n;     ///< Number of columns in the global matrix
  int mb;    ///< Row blocking factor
  int nb;    ///< Column blocking factor
  int isrc;  ///< Process row of the first row of the global matrix
  int jsrc;  ///< Process column of the first column of the global matrix
  int i;     ///< First row of the submatrix within global matrix, has to be 1
  int j;     ///< First column of the submatrix within global matrix, has to be 1
  int ld;    ///< Leading fimension of the local matrix
};
