//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

struct DLAF_descriptor make_dlaf_descriptor(int m, int n, int i, int j, int* desc) {
  struct DLAF_descriptor dlaf_desc = {m, n, desc[4], desc[5], desc[6], desc[7], i - 1, j - 1, desc[8]};
  return dlaf_desc;
}
