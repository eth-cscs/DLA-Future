//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/local/future.hpp>
#include <hpx/local/unwrap.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/eigensolver/band_to_tridiag/api.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/memory/memory_view.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
void HHReflector(const SizeType n, T& tau, T* v, T* vec) {
  DLAF_ASSERT_HEAVY(n >= 0, n);

  using dlaf::util::size_t::mul;

  // compute the reflector in-place
  lapack::larfg(n, vec, vec + 1, 1, &tau);

  // copy the HH reflector to v and set the elements annihilated by the HH transf. to 0.
  v[0] = 1.;
  blas::copy(n - 1, vec + 1, 1, v + 1, 1);
  std::memset(vec + 1, 0, mul(n - 1, sizeof(T)));
}

template <class T>
void applyHHLeftRightHerm(const SizeType n, const T tau, const T* v, T* a, const SizeType lda, T* w) {
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  blas::hemv(ColMaj, Lower, n, tau, a, lda, v, 1, 0., w, 1);

  const T tmp = -blas::dot(n, w, 1, v, 1) * tau / BaseType<T>{2.};
  blas::axpy(n, tmp, v, 1, w, 1);
  blas::her2(ColMaj, Lower, n, -1., w, 1, v, 1, a, lda);
}

template <class T>
void applyHHLeft(const SizeType m, const SizeType n, const T tau, const T* v, T* a, const SizeType lda,
                 T* w) {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  blas::gemv(ColMaj, ConjTrans, m, n, 1., a, lda, v, 1, 0., w, 1);
  blas::ger(ColMaj, m, n, -dlaf::conj(tau), v, 1, w, 1, a, lda);
}

template <class T>
void applyHHRight(const SizeType m, const SizeType n, const T tau, const T* v, T* a, const SizeType lda,
                  T* w) {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  blas::gemv(ColMaj, NoTrans, m, n, 1., a, lda, v, 1, 0., w, 1);
  blas::ger(ColMaj, m, n, -tau, w, 1, v, 1, a, lda);
}

template <class T>
class BandBlock {
  using MatrixType = Matrix<T, Device::CPU>;
  using ConstTileType = typename MatrixType::ConstTileType;

public:
  BandBlock(SizeType n, SizeType band_size)
      : size_(n), band_size_(band_size), ld_(2 * band_size_ - 1), mem_(n * (ld_ + 1)) {}

  T* ptr(SizeType offset, SizeType j) {
    DLAF_ASSERT_HEAVY(0 <= offset && offset < ld_ + 1, offset, ld_);
    DLAF_ASSERT_HEAVY(0 <= j && j < size_, j, size_);
    return mem_(j * (ld_ + 1) + offset);
  }

  SizeType ld() {
    return ld_;
  }

  void copyDiag(SizeType j, const ConstTileType& source) {
    constexpr auto General = lapack::MatrixType::General;
    constexpr auto Lower = lapack::MatrixType::Lower;

    // First set the diagonals from b+2 to 2b to 0.
    lapack::laset(General, band_size_ - 1, source.size().cols(), 0., 0., ptr(band_size_ + 1, j),
                  ld() + 1);
    // The elements are copied in the following way:
    // (a: copied with first lacpy (General), b: copied with second lacpy (Lower))
    // 6x6 tile, band_size = 3  |  2x2 tile, band_size = 3
    // a * * * * *              |
    // a a * * * *              |  b *
    // a a a * * *              |  b b
    // a a a b * *              |
    // * a a b b *              |
    // * * a b b b              |
    const auto index = std::max(SizeType{0}, source.size().cols() - band_size_);
    if (index > 0) {
      lapack::lacpy(General, band_size_ + 1, index, source.ptr(), source.ld() + 1, ptr(0, j), ld() + 1);
    }
    const auto size = std::min(band_size_, source.size().cols());
    lapack::lacpy(Lower, size, size, source.ptr({index, index}), source.ld(), ptr(0, j + index), ld());
  }

  void copyOffdiag(SizeType j, const ConstTileType& source) {
    constexpr auto General = lapack::MatrixType::General;
    constexpr auto Upper = lapack::MatrixType::Upper;
    // The elements are copied in the following way:
    // (a: copied with first lacpy (Upper), b: copied with second lacpy (General))
    // (copied when j = n)
    // 6x6 tile, band_size = 3  |  2x6 tile, band_size = 3
    // * * * a a a              |
    // * * * * a a              |  * * * a a b
    // * * * * * a              |  * * * * a b
    // * * * * * *              |
    // * * * * * *              |
    // * * * * * *              |
    const auto index = source.size().cols() - band_size_;
    const auto size = std::min(band_size_, source.size().rows());
    auto dest = ptr(band_size_, j + index);
    lapack::lacpy(Upper, size, size, source.ptr({0, index}), source.ld(), dest, ld());
    if (band_size_ > size) {
      const auto size2 = band_size_ - size;
      lapack::lacpy(General, source.size().rows(), size2, source.ptr({0, index + size}), source.ld(),
                    dest + ld() * size, ld());
    }
  }

private:
  SizeType size_;
  SizeType band_size_;
  SizeType ld_;
  memory::MemoryView<T, Device::CPU> mem_;
};

template <class T>
class SweepWorker {
public:
  SweepWorker(SizeType size, SizeType band_size)
      : size_(size), band_size_(band_size), data_(1 + 2 * band_size) {}

  SweepWorker(const SweepWorker&) = delete;
  SweepWorker(SweepWorker&&) = default;

  SweepWorker& operator=(const SweepWorker&&) = delete;
  SweepWorker& operator=(SweepWorker&&) = default;

  void startSweep(SizeType sweep, BandBlock<T>& a) {
    startSweepInternal(sweep, a);
  }

  void compactCopyToTile(matrix::Tile<T, Device::CPU>&& tile_v, TileElementIndex index) {
    tile_v(index) = tau();
    blas::copy(sizeHHR() - 1, v() + 1, 1, tile_v.ptr(index) + 1, 1);
  }

  void doStep(BandBlock<T>& a) {
    doStepFull(a);
  }

protected:
  template <class BandBlockType>
  void startSweepInternal(SizeType sweep, BandBlockType& a) {
    SizeType n = std::min(size_ - sweep - 1, band_size_);
    HHReflector(n, tau(), v(), a.ptr(1, sweep));

    setId(sweep, 0);
  }

  template <class BandBlockType>
  void doStepFull(BandBlockType& a) {
    SizeType j = firstRowHHR();
    SizeType n = sizeHHR();  // size diagonal tile and width off-diag tile
    SizeType m = std::min(band_size_, size_ - band_size_ - j);  // height off diagonal tile

    applyHHLeftRightHerm(n, tau(), v(), a.ptr(0, j), a.ld(), w());
    if (m > 0) {
      applyHHRight(m, n, tau(), v(), a.ptr(n, j), a.ld(), w());
    }
    if (m > 1) {
      HHReflector(m, tau(), v(), a.ptr(n, j));
      applyHHLeft(m, n - 1, tau(), v(), a.ptr(n - 1, j + 1), a.ld(), w());
    }
    step_ += 1;
    // Note: the sweep is completed if m <= 1.
  }

  void setId(SizeType sweep, SizeType step) {
    sweep_ = sweep;
    step_ = step;
  }

  SizeType firstRowHHR() {
    return 1 + sweep_ + step_ * band_size_;
  }

  SizeType sizeHHR() {
    return std::min(band_size_, size_ - firstRowHHR());
  }

  T& tau() noexcept {
    return *data_(0);
  }

  T* v() noexcept {
    return data_(1);
  }

  T* w() noexcept {
    return data_(1 + band_size_);
  }

  SizeType size_;
  SizeType band_size_;
  SizeType sweep_;
  SizeType step_;
  memory::MemoryView<T, Device::CPU> data_;
};

template <class T>
struct BandToTridiag<Backend::MC, Device::CPU, T> {
  static SizeType nrSweeps(SizeType size) noexcept {
    // Complex needs an extra sweep to have a real tridiagonal matrix.
    return std::is_same<T, ComplexType<T>>::value ? size - 1 : size - 2;
  }

  static SizeType nrStepsForSweep(SizeType sweep, SizeType size, SizeType band) noexcept {
    // Last Complex sweep should be handled differently.
    return sweep == size - 2 ? 1 : util::ceilDiv(size - sweep - 2, band);
  }

  // Local implementation of bandToTridiag.
  static ReturnTridiagType<T, Device::CPU> call_L(const SizeType b,
                                                  Matrix<const T, Device::CPU>& mat_a) noexcept {
    using MatrixType = Matrix<T, Device::CPU>;
    using ConstTileType = typename MatrixType::ConstTileType;
    using common::internal::vector;
    using util::ceilDiv;

    using hpx::unwrapping;
    using hpx::resource::get_num_threads;

    auto executor_hp = dlaf::getHpExecutor<Backend::MC>();

    // note: A is square and has square blocksize
    SizeType size = mat_a.size().cols();
    SizeType nrtile = mat_a.nrTiles().cols();
    SizeType nb = mat_a.blockSize().cols();

    // Need share pointer to keep the allocation until all the tasks are executed.
    auto a_ws = std::make_shared<BandBlock<T>>(size, b);

    Matrix<BaseType<T>, Device::CPU> mat_trid({2, size}, {2, nb});
    Matrix<T, Device::CPU> mat_v({size, size}, {nb, nb});
    const auto& dist_v = mat_v.distribution();

    if (size == 0) {
      return {std::move(mat_trid), std::move(mat_v)};
    }

    const auto max_deps_size = ceilDiv(size, b);
    vector<hpx::shared_future<void>> deps;
    deps.reserve(max_deps_size);

    auto copy_diag = [a_ws](SizeType j, const ConstTileType& source) { a_ws->copyDiag(j, source); };

    auto copy_offdiag = [a_ws](SizeType j, const ConstTileType& source) {
      a_ws->copyOffdiag(j, source);
    };

    // Copy the band matrix
    for (SizeType k = 0; k < nrtile; ++k) {
      hpx::shared_future<void> sf =
          hpx::dataflow(executor_hp, unwrapping(copy_diag), k * nb, mat_a.read(GlobalTileIndex{k, k}));
      if (k < nrtile - 1) {
        for (int i = 0; i < nb / b - 1; ++i) {
          deps.push_back(sf);
        }
        sf = hpx::dataflow(executor_hp, unwrapping(copy_offdiag), k * nb,
                           mat_a.read(GlobalTileIndex{k + 1, k}), sf);
        deps.push_back(sf);
      }
      else {
        while (deps.size() < max_deps_size) {
          deps.push_back(sf);
        }
      }
    }

    // Maximum size / (2b-1) sweeps can be executed in parallel.
    const auto max_workers =
        std::min(ceilDiv(size, 2 * b - 1), to_signed<SizeType>(get_num_threads("default")));
    vector<hpx::future<SweepWorker<T>>> workers(max_workers);

    auto init_sweep = [a_ws](SizeType sweep, SweepWorker<T>&& worker) {
      worker.startSweep(sweep, *a_ws);
      return hpx::make_tuple(std::move(worker), true);
    };
    auto store_tau_v = [](SweepWorker<T>&& worker, matrix::Tile<T, Device::CPU>&& tile_v,
                          TileElementIndex index) {
      worker.compactCopyToTile(std::move(tile_v), index);
      return std::move(worker);
    };
    auto cont_sweep = [a_ws](SweepWorker<T>&& worker) {
      worker.doStep(*a_ws);
      return hpx::make_tuple(std::move(worker), true);
    };

    auto copy_tridiag = [executor_hp, a_ws, &mat_trid](SizeType sweep, hpx::shared_future<void> dep) {
      auto copy_tridiag_task = [a_ws](SizeType start, SizeType n_d, SizeType n_e, auto tile_t) {
        auto inc = a_ws->ld() + 1;
        if (std::is_same<T, ComplexType<T>>::value)
          // skip imaginary part if Complex.
          inc *= 2;

        blas::copy(n_d, (BaseType<T>*) a_ws->ptr(0, start), inc, tile_t.ptr({0, 0}), tile_t.ld());
        blas::copy(n_e, (BaseType<T>*) a_ws->ptr(1, start), inc, tile_t.ptr({1, 0}), tile_t.ld());
      };

      const auto size = mat_trid.size().cols();
      const auto nb = mat_trid.blockSize().cols();
      if (sweep % nb == nb - 1 || sweep == size - 1) {
        const auto tile_index = sweep / nb;
        const auto start = tile_index * nb;
        hpx::dataflow(executor_hp, unwrapping(copy_tridiag_task), start, std::min(nb, size - start),
                      std::min(nb, size - 1 - start), mat_trid(GlobalTileIndex{0, tile_index}), dep);
      }
    };

    const auto sweeps = nrSweeps(size);
    for (SizeType sweep = 0; sweep < sweeps; ++sweep) {
      // Create the first max_workers workers and then reuse them.
      auto worker = sweep < max_workers ? hpx::make_ready_future(SweepWorker<T>(size, b))
                                        : std::move(workers[sweep % max_workers]);

      auto ret =
          hpx::split_future(hpx::dataflow(executor_hp, unwrapping(init_sweep), sweep, worker, deps[0]));
      worker = std::move(hpx::get<0>(ret));
      copy_tridiag(sweep, hpx::future<void>(std::move(hpx::get<1>(ret))));

      const auto steps = nrStepsForSweep(sweep, size, b);
      for (SizeType step = 0; step < steps; ++step) {
        auto dep_index = std::min(step + 1, deps.size() - 1);

        const GlobalElementIndex index_v((sweep / b + step) * b, sweep);

        worker = hpx::dataflow(hpx::launch::sync, unwrapping(store_tau_v), worker,
                               mat_v(dist_v.globalTileIndex(index_v)), dist_v.tileElementIndex(index_v));
        auto ret = hpx::split_future(
            hpx::dataflow(executor_hp, unwrapping(cont_sweep), worker, deps[dep_index]));
        deps[step] = hpx::future<void>(std::move(hpx::get<1>(ret)));
        worker = std::move(hpx::get<0>(ret));
      }
      // Move the Worker structure such that it can be reused in a later sweep.
      workers[sweep % max_workers] = std::move(worker);

      // Shrink the dependency vector to only include the futures generated in this sweep.
      deps.resize(steps);
    }

    // copy the last elements of the diagonals
    if (!std::is_same<T, ComplexType<T>>::value) {
      copy_tridiag(size - 2, deps[0]);
    }
    copy_tridiag(size - 1, deps[0]);

    return {std::move(mat_trid), std::move(mat_v)};
  }
};

}
}
}
