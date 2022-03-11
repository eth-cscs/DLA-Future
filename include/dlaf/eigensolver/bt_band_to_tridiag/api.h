//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
struct BackTransformationT2B {
  static void call(const SizeType band_size, Matrix<T, D>& mat_e, Matrix<const T, Device::CPU>& mat_hh);
};

template <class T>
struct BackTransformationT2B<Backend::GPU, Device::GPU, T> {
  static void call(const SizeType band_size, Matrix<T, Device::GPU>& mat_e,
                   Matrix<const T, Device::CPU>& mat_hh) {
    dlaf::internal::silenceUnusedWarningFor(band_size, mat_e, mat_hh);
    DLAF_UNIMPLEMENTED(Backend::GPU);
  }
};
}
