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

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/local/future.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/eigensolver/band_to_tridiag/api.h"
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
void applyHHLeft(const SizeType m, const SizeType n, const T tau, const T* v, T* a, const SizeType lda, T* w) {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  blas::gemv(ColMaj, ConjTrans, m, n, 1., a, lda, v, 1, 0., w, 1);
  blas::ger(ColMaj, m, n, -dlaf::conj(tau), v, 1, w, 1, a, lda);
}

template <class T>
void applyHHRight(const SizeType m, const SizeType n, const T tau, const T* v, T* a, const SizeType lda, T* w) {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  blas::gemv(ColMaj, NoTrans, m, n, 1., a, lda, v, 1, 0., w, 1);
  blas::ger(ColMaj, m, n, -tau, w, 1, v, 1, a, lda);
}

// split versions of the previous operations
template <class T>
void applyHHLeftRightHerm(const SizeType n1, const SizeType n2, const T tau, const T* v, T* a1, T* a2, const SizeType lda,
                          T* w) {
  DLAF_ASSERT_HEAVY(n1 >= 0, n1);
  DLAF_ASSERT_HEAVY(n2 >= 0, n2);
  const auto n = n1 + n2;

  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto ColMaj = blas::Layout::ColMajor;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;

  blas::hemv(ColMaj, Lower, n1, tau, a1, lda, v, 1, 0., w, 1);
  blas::gemv(ColMaj, ConjTrans, n2, n1, tau, a1 + n1, lda, v + n1, 1, 0., w, 1);
  blas::gemv(ColMaj, NoTrans, n2, n1, tau, a1 + n1, lda, v, 1, 0., w + n1, 1);
  blas::hemv(ColMaj, Lower, n2, tau, a2, lda, v + n1, 1, 0., w + n1, 1);

  const T tmp = -blas::dot(n, w, 1, v, 1) * tau / BaseType<T>{2.};
  blas::axpy(n, tmp, v, 1, w, 1);
  blas::her2(ColMaj, Lower, n1, -1., w, 1, v, 1, a1, lda);
  blas::ger(ColMaj, n2, n1, -tau, w + n1, 1, v, 1, a1 + n1, lda);
  blas::ger(ColMaj, n2, n1, -tau, v + n1, 1, w, 1, a1 + n1, lda);
  blas::her2(ColMaj, Lower, n2, -1., w + n1, 1, v + n1, 1, a2, lda);
}

template <class T>
void applyHHLeft(const SizeType m, const SizeType n1, const SizeType n2, const T tau, const T* v, T* a1, T* a2, const SizeType lda,
                 T* w) {
  applyHHLeft(m, n1, tau, v, a1, lda, w);
  applyHHLeft(m, n2, tau, v, a2, lda, w);
}

template <class T>
void applyHHRight(const SizeType m, const SizeType n1, const SizeType n2, const T tau, const T* v, T* a1, T* a2, const SizeType lda,
                  T* w) {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n1 >= 0, n1);
  DLAF_ASSERT_HEAVY(n2 >= 0, n2);

  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  blas::gemv(ColMaj, NoTrans, m, n1, 1., a1, lda, v, 1, 0., w, 1);
  blas::gemv(ColMaj, NoTrans, m, n2, 1., a2, lda, v + n1, 1, 1., w, 1);
  blas::ger(ColMaj, m, n1, -tau, w, 1, v, 1, a1, lda);
  blas::ger(ColMaj, m, n2, -tau, w, 1, v + n1, 1, a2, lda);
}

template <class T, bool dist = false>
class BandBlock {
  using MatrixType = Matrix<T, Device::CPU>;
  using ConstTileType = typename MatrixType::ConstTileType;

public:
  // Local constructor
  template <bool dist2 = dist, std::enable_if_t<!dist2 && dist == dist2, int> = 0>
  BandBlock(SizeType n, SizeType band_size)
      : size_(n), band_size_(band_size), ld_(2 * band_size_ - 1), id_(0), block_size_(n),
        mem_(n * (ld_ + 1)) {}

  // Distributed constructor
  template <bool dist2 = dist, std::enable_if_t<dist2 && dist == dist2, int> = 0>
  BandBlock(SizeType n, SizeType band_size, SizeType id, SizeType block_size)
      : size_(n), band_size_(band_size), ld_(2 * band_size_ - 1), id_(id), block_size_(block_size),
        mem_((block_size_ + band_size_) * (ld_ + 1)) {
    using util::ceilDiv;
    DLAF_ASSERT(0 <= n, n);
    // Note: band_size_ = 1 means already tridiagonal.
    DLAF_ASSERT(2 <= band_size, block_size_);
    DLAF_ASSERT(2 <= block_size_, block_size_);
    DLAF_ASSERT(block_size_ % band_size_ == 0, block_size_, band_size_);
    DLAF_ASSERT(0 <= id && id < ceilDiv(size_, block_size_), id, ceilDiv(size_, block_size_));
  }

  T* ptr(SizeType offset, SizeType j) {
    DLAF_ASSERT_HEAVY(0 <= offset && offset < ld_ + 1, offset, ld_);
    DLAF_ASSERT_HEAVY(0 <= j && j < size_, j, size_);

    if (dist) {
      return mem_(memoryIndex(j) * (ld_ + 1) + offset);
    }
    else {
      return mem_(j * (ld_ + 1) + offset);
    }
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

  SizeType nextSplit(SizeType j) {
    // TODO
    return 10;
  }


private:
  SizeType memoryIndex(SizeType j) {
    if (dist) {
      DLAF_ASSERT_HEAVY(block_size_ * id_ <= j && j < size_, j, id_, block_size_, size_);
      return (j - block_size_ * id_) % (block_size_ + band_size_);
    }
    else {
      DLAF_ASSERT_HEAVY(0 <= j && j < size_, j, size_);
      return j;
    }
  }

  SizeType size_;
  SizeType band_size_;
  SizeType ld_;

  SizeType id_;
  SizeType block_size_;

  memory::MemoryView<T, Device::CPU> mem_;
};

// template<class T>
// T* BandBlock<T, true>::ptr(SizeType offset, SizeType j) {
//}

template <class T>
using BandBlockDist = BandBlock<T, true>;

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
    // std::cout << sweep_ << ", " << step_ << std::endl;
    SizeType j = 1 + sweep_ + step_ * band_size_;
    SizeType n = std::min(band_size_, size_ - j);  // size diagonal tile and width off-diag tile
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
class SweepWorkerDist : private SweepWorker<T> {
public:
  SweepWorkerDist(SizeType size, SizeType band_size)
    :SweepWorker<T>(size, band_size) {}

  void startSweep(SizeType sweep, BandBlockDist<T>& a) {
    this->startSweepInternal(sweep, a);
  }

  void doStep(BandBlockDist<T>& a) {
    SizeType j = 1 + sweep_ + step_ * band_size_;
    SizeType n = std::min(band_size_, size_ - j);  // size diagonal tile and width off-diag tile

    const auto n1 = a.nextSplit(j);
    if (n1 > n) {
      this->doStepFull(a);
    }
    else {
      doStepSplit(a, n1);
    }
  }

private:
  void doStepSplit(BandBlockDist<T>& a, SizeType n1) {
    // std::cout << sweep_ << ", " << step_ << std::endl;
    SizeType j = 1 + sweep_ + step_ * band_size_;
    SizeType n = std::min(band_size_, size_ - j);  // size diagonal tile and width off-diag tile
    SizeType m = std::min(band_size_, size_ - band_size_ - j);  // height off diagonal tile
    const auto n2 = n - n1;

    applyHHLeftRightHerm(n1, n2, tau(), v(), a.ptr(0, j), a.ptr(0, j + n1), a.ld(), w());
    if (m > 0) {
      applyHHRight(m, n1, n2, tau(), v(), a.ptr(n, j), a.ptr(n2, j + n1), a.ld(), w());
    }
    if (m > 1) {
      HHReflector(m, tau(), v(), a.ptr(n, j));
      applyHHLeft(m, n1 - 1, n2, tau(), v(), a.ptr(n - 1, j + 1), a.ptr(n2, j + n1), a.ld(), w());
    }
    step_ += 1;
    // Note: the sweep is completed if m <= 1.
  }

  using SweepWorker<T>::size_;
  using SweepWorker<T>::band_size_;
  using SweepWorker<T>::sweep_;
  using SweepWorker<T>::step_;
  using SweepWorker<T>::tau;
  using SweepWorker<T>::v;
  using SweepWorker<T>::w;
};

template <class T>
struct BandToTridiag<Backend::MC, Device::CPU, T> {
  // Local implementation of bandToTridiag.
  static auto call_L(const SizeType b, Matrix<T, Device::CPU>& mat_a) {
    using MatrixType = Matrix<T, Device::CPU>;
    using ConstTileType = typename MatrixType::ConstTileType;
    using common::internal::vector;
    using util::ceilDiv;

    using hpx::util::unwrapping;
    using hpx::execution::parallel_executor;
    using hpx::resource::get_num_threads;
    using hpx::resource::get_thread_pool;
    using hpx::threads::thread_priority;

    parallel_executor executor_hp(&get_thread_pool("default"), thread_priority::high);
    parallel_executor executor_normal(&get_thread_pool("default"), thread_priority::default_);

    // note: A is square and has square blocksize
    SizeType size = mat_a.size().cols();
    SizeType nrtile = mat_a.nrTiles().cols();
    SizeType nb = mat_a.blockSize().cols();

    // Need share pointer to keep the allocation until all the tasks are executed.
    auto a_ws = std::make_shared<BandBlock<T>>(size, b);

    // TODO: define how to output the diags.
    vector<hpx::shared_future<vector<BaseType<T>>>> d;
    d.reserve(nrtile);
    vector<hpx::shared_future<vector<BaseType<T>>>> e;
    e.reserve(nrtile);

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

    // DEBUG:
    // hpx::wait_all(*((std::vector<hpx::shared_future<void>>*) &deps));
    // for (SizeType i = 0; i < 2 * b; ++i) {
    //   for (SizeType j = 0; j < size - i; ++j)
    //     std::cout << *a_ws->ptr(i, j) << ", ";
    //   std::cout << std::endl;
    // }

    // Maximum size / (2b-1) sweeps can be executed in parallel.
    const auto max_workers =
        std::min(ceilDiv(size, 2 * b - 1), to_signed<SizeType>(get_num_threads("default")));
    vector<hpx::future<SweepWorker<T>>> workers(max_workers);

    auto init_sweep = [a_ws](SizeType sweep, SweepWorker<T>&& worker) {
      worker.startSweep(sweep, *a_ws);
      return hpx::make_tuple(std::move(worker), true);
    };
    auto cont_sweep = [a_ws](SweepWorker<T>&& worker) {
      worker.doStep(*a_ws);
      return hpx::make_tuple(std::move(worker), true);
    };

    auto copy_tridiag = [executor_hp, size, nb, a_ws, &d, &e](SizeType sweep,
                                                              hpx::shared_future<void> dep) {
      auto copy_tridiag_task = [a_ws](SizeType start, SizeType n_d, SizeType n_e) {
        vector<BaseType<T>> db(n_d);
        vector<BaseType<T>> eb(n_e);

        auto inc = a_ws->ld() + 1;
        if (std::is_same<T, ComplexType<T>>::value)
          // skip imaginary part if Complex.
          inc *= 2;

        blas::copy(n_d, (BaseType<T>*) a_ws->ptr(0, start), inc, db.data(), 1);
        blas::copy(n_e, (BaseType<T>*) a_ws->ptr(1, start), inc, eb.data(), 1);

        return hpx::make_tuple(db, eb);
      };

      if (sweep % nb == nb - 1 || sweep == size - 1) {
        const auto start = sweep / nb * nb;
        auto ret = hpx::split_future(hpx::dataflow(executor_hp, unwrapping(copy_tridiag_task), start,
                                                   std::min(nb, size - start),
                                                   std::min(nb, size - 1 - start), dep));

        d.push_back(std::move(hpx::get<0>(ret)));
        e.push_back(std::move(hpx::get<1>(ret)));
      }
    };

    const auto sweeps = std::is_same<T, ComplexType<T>>::value ? size - 1 : size - 2;
    for (SizeType sweep = 0; sweep < sweeps; ++sweep) {
      // Create the first max_workers workers and then reuse them.
      auto worker = sweep < max_workers ? hpx::make_ready_future(SweepWorker<T>(size, b))
                                        : std::move(workers[sweep % max_workers]);

      auto ret =
          hpx::split_future(hpx::dataflow(executor_hp, unwrapping(init_sweep), sweep, worker, deps[0]));
      worker = std::move(hpx::get<0>(ret));
      copy_tridiag(sweep, hpx::future<void>(std::move(hpx::get<1>(ret))));

      const auto steps = sweep == size - 2 ? 1 : ceilDiv(size - sweep - 2, b);
      for (SizeType step = 0; step < steps; ++step) {
        auto dep_index = std::min(step + 1, deps.size() - 1);

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

    // DEBUG:
    // deps[0].wait();
    // for (SizeType i = 0; i < 2 * b; ++i) {
    //   for (SizeType j = 0; j < size - i; ++j)
    //     std::cout << *a_ws->ptr(i, j) << ", ";
    //   std::cout << std::endl;
    // }

    // TODO: define how to output the diags.
    return std::make_tuple(d, e);
  }

  // Distributed implementation of bandToTridiag.
  static auto call_L(comm::CommunicatorGrid grid, const SizeType b, Matrix<T, Device::CPU>& mat_a) {
    using MatrixType = Matrix<T, Device::CPU>;
    using ConstTileType = typename MatrixType::ConstTileType;
    using common::internal::vector;
    using util::ceilDiv;

    using hpx::util::unwrapping;
    using hpx::execution::parallel_executor;
    using hpx::resource::get_num_threads;
    using hpx::resource::get_thread_pool;
    using hpx::threads::thread_priority;

    parallel_executor executor_hp(&get_thread_pool("default"), thread_priority::high);
    parallel_executor executor_normal(&get_thread_pool("default"), thread_priority::default_);

    comm::Executor executor_mpi{};
    common::Pipeline<comm::Communicator> mpi_task_chain(grid.fullCommunicator());

    // Should be dispatched to local implementation if (1x1) grid.
    DLAF_ASSERT(grid.size() != comm::Size2D(1, 1), grid);

    // note: A is square and has square blocksize
    SizeType size = mat_a.size().cols();
    SizeType nrtile = mat_a.nrTiles().cols();
    SizeType nb = mat_a.blockSize().cols();
    auto& dist_a = mat_a.distribution();

    const auto rank = grid.rankFullCommunicator(grid.rank());
    const auto ranks = static_cast<comm::IndexT_MPI>(grid.size().linear_size());
    const auto prev_rank = (rank == 0 ? ranks - 1 : rank - 1);
    const auto next_rank = (rank + 1 == ranks ? 0 : rank + 1);

    SizeType tiles_per_block = 1;
    matrix::Distribution dist({1, size}, {1, nb * tiles_per_block}, {1, ranks}, {0, rank}, {0, 0});

    // Need share pointer to keep the allocation until all the tasks are executed.
    vector<std::shared_ptr<BandBlock<T, true>>> a_ws;
    for (SizeType i = 0; i < dist.localNrTiles().cols(); ++i) {
      a_ws.emplace_back(
          std::make_shared<BandBlock<T, true>>(size, b, rank + i * ranks, nb * tiles_per_block));
    }

    // TODO: define how to output the diags.
    vector<hpx::shared_future<vector<BaseType<T>>>> d;
    d.reserve(nrtile);
    vector<hpx::shared_future<vector<BaseType<T>>>> e;
    e.reserve(nrtile);

    const auto max_deps_size = ceilDiv(dist.localSize().cols(), b);
    vector<hpx::shared_future<void>> deps;
    deps.reserve(max_deps_size);

    auto copy_diag = [](std::shared_ptr<BandBlock<T, true>> a_block, SizeType j, const ConstTileType& source) { (a_block)->copyDiag(j, source); };

    auto copy_offdiag = [](std::shared_ptr<BandBlock<T, true>> a_block, SizeType j, const ConstTileType& source) {
      a_block->copyOffdiag(j, source);
    };

    // TODO send receive and distribution
    // Copy the band matrix
    for (SizeType k = 0; k < nrtile; ++k) {
      const auto id_block = k / tiles_per_block;
      const GlobalTileIndex index_diag{k, k};
      const GlobalTileIndex index_offdiag{k + 1, k};
      const auto rank_block = dist.rankGlobalTile<Coord::Col>(id_block);
      const auto rank_diag = grid.rankFullCommunicator(dist_a.rankGlobalTile(index_diag));
      const auto rank_offdiag = (k == nrtile - 1 ? -1 : grid.rankFullCommunicator(dist_a.rankGlobalTile(index_offdiag)));

      if (rank == rank_block) {
        auto diag_tile = (rank == rank_diag
                              ? mat_a.read(index_diag)
                              : comm::scheduleRecvAlloc<T, Device::CPU>(executor_mpi, dist_a.tileSize(index_diag), rank_diag, 0, mpi_task_chain()));

        const auto id_block_local = dist.localTileFromGlobalTile<Coord::Col>(id_block);

        hpx::shared_future<void> sf =
            hpx::dataflow(executor_hp, unwrapping(copy_diag), a_ws[id_block_local], k * nb, diag_tile);

        if (k < nrtile - 1) {
          for (int i = 0; i < nb / b - 1; ++i) {
            deps.push_back(sf);
          }
          auto offdiag_tile =
              (rank == rank_offdiag
                   ? mat_a.read(index_offdiag)
                   : comm::scheduleRecvAlloc<T, Device::CPU>(executor_mpi, dist_a.tileSize(index_offdiag), rank_offdiag, 0, mpi_task_chain()));

          sf = hpx::dataflow(executor_hp, unwrapping(copy_offdiag), a_ws[id_block_local], k * nb, offdiag_tile, sf);
          deps.push_back(sf);
        }
        else {
          while (deps.size() < max_deps_size) {
            deps.push_back(sf);
          }
        }
      }
      else {
        if (rank == rank_diag) {
          scheduleSend(executor_mpi, mat_a.read(index_diag), rank_block, 0, mpi_task_chain());
        }
        if (k < nrtile - 1) {
          if (rank == rank_offdiag) {
            scheduleSend(executor_mpi, mat_a.read(index_offdiag), rank_block, 0, mpi_task_chain());
          }
        }
      }
    }

    // DEBUG:
    hpx::wait_all(*((std::vector<hpx::shared_future<void>>*) &deps));
    for (SizeType i = 0; i < 2 * b; ++i) {
       for (SizeType jj = 0; jj < dist.nrTiles().cols(); ++jj) {
         for (SizeType j = 0; j < dist.tileSize(GlobalTileIndex{0, jj}).cols(); ++j) {
           if (rank == dist.rankGlobalTile({0, jj}).col()){
             std::cout << *a_ws[dist.localTileIndex({0, jj}).col()]->ptr(i, jj * dist.blockSize().cols() + j) << ", ";
           }
         }
       }
       std::cout << std::endl;
    }

    // Maximum block_size / (2b-1) sweeps per block can be executed in parallel.
    const auto max_workers =
        dist.localNrTiles().cols() * std::min(ceilDiv(dist.blockSize().cols(), 2 * b - 1), to_signed<SizeType>(get_num_threads("default")));
    vector<hpx::future<SweepWorkerDist<T>>> workers(max_workers);

    auto init_sweep = [a_ws](std::shared_ptr<BandBlock<T, true>> a_block, SizeType sweep, SweepWorkerDist<T>&& worker) {
      worker.startSweep(sweep, *a_block);
      return hpx::make_tuple(std::move(worker), true);
    };
    auto cont_sweep = [a_ws](std::shared_ptr<BandBlock<T, true>> a_block, SweepWorkerDist<T>&& worker) {
      worker.doStep(*a_block);
      return hpx::make_tuple(std::move(worker), true);
    };

    /*
    auto copy_tridiag = [executor_hp, size, nb, a_ws, &d, &e](SizeType sweep,
                                                              hpx::shared_future<void> dep) {
      auto copy_tridiag_task = [a_ws](SizeType start, SizeType n_d, SizeType n_e) {
        vector<BaseType<T>> db(n_d);
        vector<BaseType<T>> eb(n_e);

        auto inc = a_ws->ld() + 1;
        if (std::is_same<T, ComplexType<T>>::value)
          //skip imaginary part if Complex.
          inc *= 2;

        blas::copy(n_d, (BaseType<T>*) a_ws->ptr(0, start), inc, db.data(), 1);
        blas::copy(n_e, (BaseType<T>*) a_ws->ptr(1, start), inc, eb.data(), 1);

        return hpx::make_tuple(db, eb);
      };

      if (sweep % nb == nb - 1 || sweep == size - 1) {
        const auto start = sweep / nb * nb;
        auto ret = hpx::split_future(hpx::dataflow(executor_hp, unwrapping(copy_tridiag_task), start,
                                                   std::min(nb, size - start),
                                                   std::min(nb, size - 1 - start), dep));

        d.push_back(std::move(hpx::get<0>(ret)));
        e.push_back(std::move(hpx::get<1>(ret)));
      }
    };
    */


    // TODO send/recv "to left"
    // TODO deps
    const auto sweeps = std::is_same<T, ComplexType<T>>::value ? size - 1 : size - 2;
    for (SizeType sweep = 0; sweep < sweeps; ++sweep) {
      const auto steps = sweep == size - 2 ? 1 : ceilDiv(size - sweep - 2, b);
      for (SizeType id_block = 0; id_block * dist.blockSize().cols() / b < steps; ++id_block) {
        const auto rank_block = dist.rankGlobalTile<Coord::Col>(id_block);
        const auto id_worker = 0;
        // Create the first max_workers workers and then reuse them.
        auto worker = id_worker < max_workers ? hpx::make_ready_future(SweepWorkerDist<T>(size, b))
            : std::move(workers[id_worker % max_workers]);

        if (rank == rank_block) {
          if (id_block == 0) {
            auto ret =
                hpx::split_future(hpx::dataflow(executor_hp, unwrapping(init_sweep), a_ws[0], sweep, worker, deps[0]));
            worker = std::move(hpx::get<0>(ret));
            //copy_tridiag(sweep, hpx::future<void>(std::move(hpx::get<1>(ret))));
          }
          else {
            // TODO Recv worker
          }

          // TODO
          const auto steps = sweep == size - 2 ? 1 : ceilDiv(size - sweep - 2, b);
          for (SizeType step = 0; step < steps; ++step) {
            auto dep_index = std::min(step + 1, deps.size() - 1);

            auto ret = hpx::split_future(
                hpx::dataflow(executor_hp, unwrapping(cont_sweep), a_ws[id_block], worker, deps[dep_index]));
            deps[step] = hpx::future<void>(std::move(hpx::get<1>(ret)));
            worker = std::move(hpx::get<0>(ret));
          }

          //if (id_block != last_block)
            // TODO: send worker
        }
        // Move the Worker structure such that it can be reused in a later sweep.
        workers[sweep % max_workers] = std::move(worker);
      }

      // Shrink the dependency vector to only include the futures generated in this sweep.
      deps.resize(steps);
    }

    // copy the last elements of the diagonals
    //if (!std::is_same<T, ComplexType<T>>::value) {
    //  copy_tridiag(size - 2, deps[0]);
    //}
    //copy_tridiag(size - 1, deps[0]);

    // DEBUG:
    // deps[0].wait();
    // for (SizeType i = 0; i < 2 * b; ++i) {
    //   for (SizeType j = 0; j < size - i; ++j)
    //     std::cout << *a_ws->ptr(i, j) << ", ";
    //   std::cout << std::endl;
    // }

    // TODO: define how to output the diags.
    return std::make_tuple(d, e);
  }
};

}
}
}
