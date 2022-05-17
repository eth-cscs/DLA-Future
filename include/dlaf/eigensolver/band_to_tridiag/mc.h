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

#include "dlaf/eigensolver/band_to_tridiag/api.h"

#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/eigensolver/band_to_tridiag/api.h"
#include "dlaf/lapack/gpu/lacpy.h"
#include "dlaf/lapack/gpu/laset.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/traits.h"

namespace dlaf::eigensolver::internal {

template <class T>
void HHReflector(const SizeType n, T& tau, T* v, T* vec) noexcept {
  DLAF_ASSERT_HEAVY(n >= 0, n);

  using dlaf::util::size_t::mul;

  // compute the reflector in-place
  lapack::larfg(n, vec, vec + 1, 1, &tau);

  // copy the HH reflector to v and set the elements annihilated by the HH transf. to 0.
  v[0] = 1.;
  blas::copy(n - 1, vec + 1, 1, v + 1, 1);
  std::fill(vec + 1, vec + n, T{});
}

template <class T>
void applyHHLeftRightHerm(const SizeType n, const T tau, const T* v, T* a, const SizeType lda,
                          T* w) noexcept {
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
                 T* w) noexcept {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  blas::gemv(ColMaj, ConjTrans, m, n, 1., a, lda, v, 1, 0., w, 1);
  blas::ger(ColMaj, m, n, -dlaf::conj(tau), v, 1, w, 1, a, lda);
}

template <class T>
void applyHHRight(const SizeType m, const SizeType n, const T tau, const T* v, T* a, const SizeType lda,
                  T* w) noexcept {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  blas::gemv(ColMaj, NoTrans, m, n, 1., a, lda, v, 1, 0., w, 1);
  blas::ger(ColMaj, m, n, -tau, w, 1, v, 1, a, lda);
}

template <class T>
struct BandBlock {
  BandBlock(SizeType n, SizeType band_size)
      : size_(n), band_size_(band_size), ld_(2 * band_size_ - 1), mem_(n * (ld_ + 1)) {}

  T* ptr(SizeType offset, SizeType j) noexcept {
    DLAF_ASSERT_HEAVY(0 <= offset && offset < ld_ + 1, offset, ld_);
    DLAF_ASSERT_HEAVY(0 <= j && j < size_, j, size_);
    return mem_(j * (ld_ + 1) + offset);
  }

  SizeType ld() const noexcept {
    return ld_;
  }

  template <Device D, class Sender>
  auto copyDiag(SizeType j, Sender source) noexcept {
    using dlaf::internal::transform;
    namespace ex = pika::execution::experimental;

    constexpr auto B = dlaf::matrix::internal::CopyBackend_v<D, Device::CPU>;

    if constexpr (D == Device::CPU) {
      return transform(
          dlaf::internal::Policy<B>(pika::threads::thread_priority::high),
          [=](const matrix::Tile<const T, D>& source) {
            constexpr auto General = blas::Uplo::General;
            constexpr auto Lower = blas::Uplo::Lower;

            // First set the diagonals from b+2 to 2b to 0.
            lapack::laset(General, band_size_ - 1, source.size().cols(), T(0), T(0),
                          ptr(band_size_ + 1, j), ld() + 1);
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
              lapack::lacpy(General, band_size_ + 1, index, source.ptr(), source.ld() + 1, ptr(0, j),
                            ld() + 1);
            }
            const auto size = std::min(band_size_, source.size().cols());
            lapack::lacpy(Lower, size, size, source.ptr({index, index}), source.ld(), ptr(0, j + index),
                          ld());
          },
          std::move(source));
    }
#ifdef DLAF_WITH_CUDA
    else if constexpr (D == Device::GPU) {
      DLAF_ASSERT_HEAVY(isAccessibleFromGPU(), "BandBlock memory should be accessible from GPU");
      return transform(
          dlaf::internal::Policy<B>(pika::threads::thread_priority::high),
          [=](const matrix::Tile<const T, D>& source, cudaStream_t stream) {
            constexpr auto General = blas::Uplo::General;
            constexpr auto Lower = blas::Uplo::Lower;

            // First set the diagonals from b+2 to 2b to 0.
            gpulapack::laset(General, band_size_ - 1, source.size().cols(), T(0), T(0),
                             ptr(band_size_ + 1, j), ld() + 1, stream);
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
              gpulapack::lacpy(General, band_size_ + 1, index, source.ptr(), source.ld() + 1, ptr(0, j),
                               ld() + 1, stream);
            }
            const auto size = std::min(band_size_, source.size().cols());
            gpulapack::lacpy(Lower, size, size, source.ptr({index, index}), source.ld(),
                             ptr(0, j + index), ld(), stream);
          },
          std::move(source));
    }
#endif
    else {
      return DLAF_UNREACHABLE(decltype(ex::just()));
    }
  }

  template <Device D, class Sender>
  auto copyOffDiag(const SizeType j, Sender source) noexcept {
    using dlaf::internal::transform;

    namespace ex = pika::execution::experimental;

    constexpr auto B = dlaf::matrix::internal::CopyBackend_v<D, Device::CPU>;

    if constexpr (D == Device::CPU) {
      return transform(
          dlaf::internal::Policy<B>(pika::threads::thread_priority::high),
          [=](const matrix::Tile<const T, D>& source) {
            constexpr auto General = blas::Uplo::General;
            constexpr auto Upper = blas::Uplo::Upper;
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
            ::lapack::lacpy(Upper, size, size, source.ptr({0, index}), source.ld(), dest, ld());
            if (band_size_ > size) {
              const auto size2 = band_size_ - size;
              ::lapack::lacpy(General, source.size().rows(), size2, source.ptr({0, index + size}),
                              source.ld(), dest + ld() * size, ld());
            }
          },
          std::move(source));
    }
#ifdef DLAF_WITH_CUDA
    else if constexpr (D == Device::GPU) {
      DLAF_ASSERT_HEAVY(isAccessibleFromGPU(), "BandBlock memory should be accessible from GPU");
      return transform(
          dlaf::internal::Policy<B>(pika::threads::thread_priority::high),
          [=](const matrix::Tile<const T, D>& source, cudaStream_t stream) {
            constexpr auto General = blas::Uplo::General;
            constexpr auto Upper = blas::Uplo::Upper;
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
            gpulapack::lacpy(Upper, size, size, source.ptr({0, index}), source.ld(), dest, ld(), stream);
            if (band_size_ > size) {
              const auto size2 = band_size_ - size;
              gpulapack::lacpy(General, source.size().rows(), size2, source.ptr({0, index + size}),
                               source.ld(), dest + ld() * size, ld(), stream);
            }
          },
          std::move(source));
    }
#endif
    else {
      return DLAF_UNREACHABLE(decltype(ex::just()));
    }
  }

private:
#ifdef DLAF_WITH_CUDA
  bool isAccessibleFromGPU() const {
    cudaPointerAttributes attrs;
    DLAF_CUDA_CHECK_ERROR(cudaPointerGetAttributes(&attrs, mem_()));
    return cudaMemoryTypeUnregistered != attrs.type;
  }
#endif

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

  void startSweep(SizeType sweep, BandBlock<T>& a) noexcept {
    startSweepInternal(sweep, a);
  }

  void compactCopyToTile(matrix::Tile<T, Device::CPU>& tile_v, TileElementIndex index) const noexcept {
    tile_v(index) = tau();
    blas::copy(sizeHHR() - 1, v() + 1, 1, tile_v.ptr(index) + 1, 1);
  }

  void doStep(BandBlock<T>& a) noexcept {
    doStepFull(a);
  }

protected:
  template <class BandBlockType>
  void startSweepInternal(SizeType sweep, BandBlockType& a) noexcept {
    SizeType n = std::min(size_ - sweep - 1, band_size_);
    HHReflector(n, tau(), v(), a.ptr(1, sweep));

    setId(sweep, 0);
  }

  template <class BandBlockType>
  void doStepFull(BandBlockType& a) noexcept {
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

  void setId(SizeType sweep, SizeType step) noexcept {
    sweep_ = sweep;
    step_ = step;
  }

  SizeType firstRowHHR() const noexcept {
    return 1 + sweep_ + step_ * band_size_;
  }

  SizeType sizeHHR() const noexcept {
    return std::min(band_size_, size_ - firstRowHHR());
  }

  T& tau() noexcept {
    return *data_(0);
  }
  const T& tau() const noexcept {
    return *data_(0);
  }

  T* v() noexcept {
    return data_(1);
  }
  const T* v() const noexcept {
    return data_(1);
  }

  T* w() noexcept {
    return data_(1 + band_size_);
  }

  SizeType size_;
  SizeType band_size_;
  SizeType sweep_ = 0;
  SizeType step_ = 0;
  memory::MemoryView<T, Device::CPU> data_;
};

template <Device D, class T>
TridiagResult<T, Device::CPU> BandToTridiag<Backend::MC, D, T>::call_L(
    const SizeType b, Matrix<const T, D>& mat_a) noexcept {
  // Note on the algorithm and dependency tracking:
  // The algorithm is composed by n-2 (real) or n-1 (complex) sweeps:
  // The i-th sweep is initialized by init_sweep which act on the i-th column of the band matrix.
  // Then the sweep continues applying steps.
  // The j-th step acts on the columns [i+1 + j * b, i+1 + (j+1) * b)
  // The steps in the same sweep has to be executed in order and the dependencies are managed by the
  // worker pipelines. The deps vector is used to set the dependencies among two different sweeps.
  //
  // assuming b = 4 and nb = 8 (i.e each task applies two steps):
  // Copy of band: A A A A B B B B C C C C D D D D E ...
  //                   deps[0]    |    deps[1]    | ...
  // Sweep 0       I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
  //                |    deps[0]    |    deps[1]    | ...
  // Sweep 1         I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
  //                  |    deps[0]    |    deps[1]    | ...
  // Sweep 2           I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3
  //                    ...
  // Note: j-th task (in this case 2*j-th and 2*j+1-th steps) depends explicitly only on deps[j+1],
  //       as the pipeline dependency on j-1-th task (or sweep_init for j=0) implies a dependency on
  //       deps[j] as well.

  using common::Pipeline;
  using common::PromiseGuard;
  using common::internal::vector;
  using util::ceilDiv;

  using pika::resource::get_num_threads;

  namespace ex = pika::execution::experimental;

  // note: A is square and has square blocksize
  const SizeType size = mat_a.size().cols();
  const SizeType nrtiles = mat_a.nrTiles().cols();
  const SizeType nb = mat_a.blockSize().cols();

  // Need share pointer to keep the allocation until all the tasks are executed.
  auto a_ws = std::make_shared<BandBlock<T>>(size, b);

  Matrix<BaseType<T>, Device::CPU> mat_trid({size, 2}, {nb, 2});
  Matrix<T, Device::CPU> mat_v({size, size}, {nb, nb});
  const auto& dist_v = mat_v.distribution();

  if (size == 0) {
    return {std::move(mat_trid), std::move(mat_v)};
  }

  const auto max_deps_size = nrtiles;
  vector<pika::execution::experimental::any_sender<>> deps;
  deps.reserve(max_deps_size);

  auto copy_diag = [a_ws](SizeType j, auto source) {
    return a_ws->template copyDiag<D>(j, std::move(source));
  };

  auto copy_offdiag = [a_ws](SizeType j, auto source) {
    return a_ws->template copyOffDiag<D>(j, std::move(source));
  };

  // Copy the band matrix
  for (SizeType k = 0; k < nrtiles; ++k) {
    auto sf = copy_diag(k * nb, mat_a.read_sender(GlobalTileIndex{k, k})) | ex::split();
    if (k < nrtiles - 1) {
      auto sf2 = copy_offdiag(k * nb, ex::when_all(std::move(sf),
                                                   mat_a.read_sender(GlobalTileIndex{k + 1, k}))) |
                 ex::split();
      deps.push_back(std::move(sf2));
    }
    else {
      deps.push_back(sf);
    }
  }

  // Maximum size / (2b-1) sweeps can be executed in parallel, however due to task combination it reduces
  // to size / (2nb-1).
  const auto max_workers =
      std::min(ceilDiv(size, 2 * nb - 1), 2 * to_SizeType(get_num_threads("default")));

  vector<Pipeline<SweepWorker<T>>> workers;
  workers.reserve(max_workers);
  for (SizeType i = 0; i < max_workers; ++i)
    workers.emplace_back(SweepWorker<T>(size, b));

  auto init_sweep = [a_ws](SizeType sweep, PromiseGuard<SweepWorker<T>> worker) {
    worker.ref().startSweep(sweep, *a_ws);
  };
  auto cont_sweep = [a_ws, b](SizeType nr_steps, PromiseGuard<SweepWorker<T>> worker,
                              matrix::Tile<T, Device::CPU>&& tile_v, TileElementIndex index) {
    for (SizeType j = 0; j < nr_steps; ++j) {
      worker.ref().compactCopyToTile(tile_v, index + TileElementSize(j * b, 0));
      worker.ref().doStep(*a_ws);
    }
  };

  auto policy_hp = dlaf::internal::Policy<Backend::MC>(pika::threads::thread_priority::high);
  auto copy_tridiag = [policy_hp, a_ws, &mat_trid](SizeType sweep, auto&& dep) {
    auto copy_tridiag_task = [a_ws](SizeType start, SizeType n_d, SizeType n_e, auto tile_t) {
      auto inc = a_ws->ld() + 1;
      if (isComplex_v<T>)
        // skip imaginary part if Complex.
        inc *= 2;

      blas::copy(n_d, (BaseType<T>*) a_ws->ptr(0, start), inc, tile_t.ptr({0, 0}), 1);
      blas::copy(n_e, (BaseType<T>*) a_ws->ptr(1, start), inc, tile_t.ptr({0, 1}), 1);
    };

    const auto size = mat_trid.size().rows();
    const auto nb = mat_trid.blockSize().rows();
    if (sweep % nb == nb - 1 || sweep == size - 1) {
      const auto tile_index = sweep / nb;
      const auto start = tile_index * nb;
      dlaf::internal::whenAllLift(start, std::min(nb, size - start), std::min(nb, size - 1 - start),
                                  mat_trid.readwrite_sender(GlobalTileIndex{tile_index, 0}),
                                  std::forward<decltype(dep)>(dep)) |
          dlaf::internal::transformDetach(policy_hp, copy_tridiag_task);
    }
    else {
      ex::start_detached(std::forward<decltype(dep)>(dep));
    }
  };

  const SizeType steps_per_task = nb / b;
  const SizeType sweeps = nrSweeps<T>(size);
  for (SizeType sweep = 0; sweep < sweeps; ++sweep) {
    // Create the first max_workers workers and then reuse them.
    auto& w_pipeline = workers[sweep % max_workers];

    auto dep = dlaf::internal::whenAllLift(sweep, w_pipeline(), deps[0]) |
               dlaf::internal::transform(policy_hp, init_sweep);
    copy_tridiag(sweep, std::move(dep));

    const SizeType steps = nrStepsForSweep(sweep, size, b);

    SizeType last_step = 0;
    for (SizeType step = 0; step < steps;) {
      // First task might apply less steps to align with the boundaries of the HHR tile v.
      SizeType nr_steps = steps_per_task - (step == 0 ? (sweep % nb) / b : 0);
      // Last task only applies the remaining steps
      nr_steps = std::min(nr_steps, steps - step);

      auto dep_index = std::min(ceilDiv(step + nr_steps, nb / b), deps.size() - 1);

      const GlobalElementIndex index_v((sweep / b + step) * b, sweep);

      deps[ceilDiv(step, nb / b)] =
          dlaf::internal::whenAllLift(nr_steps, w_pipeline(),
                                      mat_v.readwrite_sender(dist_v.globalTileIndex(index_v)),
                                      dist_v.tileElementIndex(index_v), deps[dep_index]) |
          dlaf::internal::transform(policy_hp, cont_sweep) | ex::split();

      last_step = step;
      step += nr_steps;
    }

    // Shrink the dependency vector to only include the futures generated in this sweep.
    deps.resize(ceilDiv(last_step, nb / b) + 1);
  }

  // copy the last elements of the diagonals
  if (!isComplex_v<T>) {
    // only needed for real types as they don't perform sweep size-2
    copy_tridiag(size - 2, deps[0]);
  }
  copy_tridiag(size - 1, std::move(deps[0]));

  return {std::move(mat_trid), std::move(mat_v)};
}

}
