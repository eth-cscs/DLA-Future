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

#include <cmath>
#include <vector>

#include "dlaf/blas/tile.h"
#include "dlaf/common/vector.h"
#include "dlaf/eigensolver/backtransformation.h"
#include "dlaf/eigensolver/band_to_tridiag.h"
#include "dlaf/eigensolver/bt_band_to_tridiag.h"
#include "dlaf/eigensolver/reduction_to_band.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf/eigensolver/eigensolver/api.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
struct Eigensolver<Backend::MC, Device::CPU, T> {
  static ReturnEigensolverType<T, Device::CPU> call(blas::Uplo uplo, Matrix<T, Device::CPU>& mat_a) {
    using common::internal::vector;

    const SizeType size = mat_a.size().rows();

    // need uplo check as reduction to band doesn't have the uplo argument yet.
    if (uplo != blas::Uplo::Lower)
      DLAF_UNIMPLEMENTED(uplo);

    auto taus = reductionToBand<Backend::MC>(mat_a);
    auto ret = bandToTridiag<Backend::MC>(uplo, mat_a.blockSize().rows(), mat_a);

    auto& mat_trid = ret.tridiagonal;
    vector<BaseType<T>> d(size);
    vector<BaseType<T>> e(size);

    // Synchronize mat_trid and copy tile by tile.
    for (SizeType j = 0; j < mat_a.nrTiles().cols(); ++j) {
      auto tile_sf = mat_trid.read(GlobalTileIndex(0, j));
      auto& tile = tile_sf.get();
      auto start = j * mat_a.blockSize().cols();
      blas::copy(tile.size().cols(), tile.ptr({0, 0}), tile.ld(), &d[start], 1);
      blas::copy(tile.size().cols(), tile.ptr({1, 0}), tile.ld(), &e[start], 1);
    }

    // mat_e ia allocated in lapack layout to be able to call stemr directly on it.
    SizeType lde = std::max<SizeType>(1, size);
    auto distr_a = mat_a.distribution();
    auto layout = matrix::colMajorLayout(distr_a, lde);
    Matrix<T, Device::CPU> mat_e(distr_a, layout);
    auto ptr_e = mat_e(GlobalTileIndex(0, 0)).get().ptr();

    // Note I'm using mrrr instead of divide & conquer as
    // mrrr is more suitable for a single core task.
    int64_t tmp;
    vector<BaseType<T>> w(size);
    vector<int64_t> isuppz(2 * std::max<SizeType>(1, size));
    bool tryrac = false;
    lapack::stemr(lapack::Job::Vec, lapack::Range::All, size, d.data(), e.data(), 0, 0, 0, 0, &tmp,
                  w.data(), ptr_e, lde, size, isuppz.data(), &tryrac);

    // Note: no sync needed here as next tasks are only scheduled
    //       after the completion of stemr.

    backTransformationBandToTridiag<Backend::MC>(mat_e, ret.hh_reflectors);
    auto taus_ = (common::internal::vector<pika::shared_future<common::internal::vector<T>>>*) &taus;
    backTransformation<Backend::MC>(mat_e, mat_a, *taus_);

    return {std::move(w), std::move(mat_e)};
  }
};

/// ---- ETI
#define DLAF_EIGENSOLVER_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct Eigensolver<Backend::MC, Device::CPU, DATATYPE>;

DLAF_EIGENSOLVER_MC_ETI(extern, float)
DLAF_EIGENSOLVER_MC_ETI(extern, double)
DLAF_EIGENSOLVER_MC_ETI(extern, std::complex<float>)
DLAF_EIGENSOLVER_MC_ETI(extern, std::complex<double>)

}
}
}
