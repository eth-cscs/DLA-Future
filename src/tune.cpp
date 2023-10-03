//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/init.h>
#include <dlaf/tune.h>

namespace dlaf {

TuneParameters& getTuneParameters() {
  static TuneParameters params;
  return params;
}

std::ostream& operator<<(std::ostream& os, const TuneParameters& params) {
  os << "  red2band_panel_nworkers = " << params.red2band_panel_nworkers << std::endl;
  os << "  red2band_barrier_busy_wait_us = " << params.red2band_barrier_busy_wait_us << std::endl;
  os << "  tridiag_rank1_nworkers = " << params.tridiag_rank1_nworkers << std::endl;
  os << "  tridiag_rank1_barrier_busy_wait_us = " << params.tridiag_rank1_barrier_busy_wait_us
     << std::endl;
  os << "  eigensolver_min_band = " << params.eigensolver_min_band << std::endl;
  os << "  band_to_tridiag_1d_block_size_base = " << params.band_to_tridiag_1d_block_size_base
     << std::endl;
  os << "  bt_band_to_tridiag_hh_apply_group_size = " << params.bt_band_to_tridiag_hh_apply_group_size
     << std::endl;
  return os;
}

}
