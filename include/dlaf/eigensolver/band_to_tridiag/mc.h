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

#include <pika/execution.hpp>
#include <pika/semaphore.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/common/vector.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/kernels.h>
#include <dlaf/eigensolver/band_to_tridiag/api.h>
#include <dlaf/eigensolver/internal/get_1d_block_size.h>
#include <dlaf/lapack/gpu/lacpy.h>
#include <dlaf/lapack/gpu/laset.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/traits.h>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

namespace dlaf::eigensolver::internal {

template <class T>
void HH_reflector(const SizeType n, T& tau, T* v, T* vec) noexcept {
  DLAF_ASSERT_HEAVY(n >= 0, n);

  using dlaf::util::size_t::mul;

  // compute the reflector in-place
  common::internal::SingleThreadedBlasScope single;
  lapack::larfg(n, vec, vec + 1, 1, &tau);

  // copy the HH reflector to v and set the elements annihilated by the HH transf. to 0.
  v[0] = 1.;
  blas::copy(n - 1, vec + 1, 1, v + 1, 1);
  std::fill(vec + 1, vec + n, T{});
}

template <class T>
void apply_HH_left_right_herm(const SizeType n, const T tau, const T* v, T* a, const SizeType lda,
                              T* w) noexcept {
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  common::internal::SingleThreadedBlasScope single;

  blas::hemv(ColMaj, Lower, n, tau, a, lda, v, 1, 0., w, 1);

  const T tmp = -blas::dot(n, w, 1, v, 1) * tau / BaseType<T>{2.};
  blas::axpy(n, tmp, v, 1, w, 1);
  blas::her2(ColMaj, Lower, n, -1., w, 1, v, 1, a, lda);
}

template <class T>
void apply_HH_left(const SizeType m, const SizeType n, const T tau, const T* v, T* a, const SizeType lda,
                   T* w) noexcept {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  common::internal::SingleThreadedBlasScope single;

  blas::gemv(ColMaj, ConjTrans, m, n, 1., a, lda, v, 1, 0., w, 1);
  blas::ger(ColMaj, m, n, -dlaf::conj(tau), v, 1, w, 1, a, lda);
}

template <class T>
void apply_HH_right(const SizeType m, const SizeType n, const T tau, const T* v, T* a,
                    const SizeType lda, T* w) noexcept {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n >= 0, n);

  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  common::internal::SingleThreadedBlasScope single;

  blas::gemv(ColMaj, NoTrans, m, n, 1., a, lda, v, 1, 0., w, 1);
  blas::ger(ColMaj, m, n, -tau, w, 1, v, 1, a, lda);
}

// As the compact band matrix is stored in a circular buffer it might happen that the operations
// have to act on a matrix which lays on the last columns and first columns of the buffer.
// The following versions take care of this case.
template <class T>
void apply_HH_left_right_herm(const SizeType n1, const SizeType n2, const T tau, const T* v, T* a1,
                              T* a2, const SizeType lda, T* w) {
  DLAF_ASSERT_HEAVY(n1 > 0, n1);
  DLAF_ASSERT_HEAVY(n2 > 0, n2);
  const auto n = n1 + n2;

  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto ColMaj = blas::Layout::ColMajor;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;

  common::internal::SingleThreadedBlasScope single;

  blas::hemv(ColMaj, Lower, n1, tau, a1, lda, v, 1, 0., w, 1);
  blas::hemv(ColMaj, Lower, n2, tau, a2, lda, v + n1, 1, 0., w + n1, 1);
  blas::gemv(ColMaj, ConjTrans, n2, n1, tau, a1 + n1, lda, v + n1, 1, 1., w, 1);
  blas::gemv(ColMaj, NoTrans, n2, n1, tau, a1 + n1, lda, v, 1, 1., w + n1, 1);

  const T tmp = -blas::dot(n, w, 1, v, 1) * tau / BaseType<T>{2.};
  blas::axpy(n, tmp, v, 1, w, 1);
  blas::her2(ColMaj, Lower, n1, -1., w, 1, v, 1, a1, lda);
  blas::ger(ColMaj, n2, n1, -1., w + n1, 1, v, 1, a1 + n1, lda);
  blas::ger(ColMaj, n2, n1, -1., v + n1, 1, w, 1, a1 + n1, lda);
  blas::her2(ColMaj, Lower, n2, -1., w + n1, 1, v + n1, 1, a2, lda);
}

template <class T>
void apply_HH_left(const SizeType m, const SizeType n1, const SizeType n2, const T tau, const T* v,
                   T* a1, T* a2, const SizeType lda, T* w) {
  apply_HH_left(m, n1, tau, v, a1, lda, w);
  apply_HH_left(m, n2, tau, v, a2, lda, w);
}

template <class T>
void apply_HH_right(const SizeType m, const SizeType n1, const SizeType n2, const T tau, const T* v,
                    T* a1, T* a2, const SizeType lda, T* w) {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n1 > 0, n1);
  DLAF_ASSERT_HEAVY(n2 > 0, n2);

  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ColMaj = blas::Layout::ColMajor;

  common::internal::SingleThreadedBlasScope single;

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
        mem_size_col_(n), mem_(mem_size_col_ * (ld_ + 1)) {}

  // Distributed constructor
  // Note on size of the buffers:
  // Due to the algorithm structure each block needs maximum two extra columns of space:
  // one for the extra column that the dependecies allow (see Fig 1), and one to safely schedule
  // the next receive.
  // However, block 0 needs extra block_size rows to store the diagonal and offdiagonal elements of the
  // tridiagonal matrix before they are copied by copy_tridiag.
  // (The copy is performed in chunks of block_size columns.)
  //
  // Fig 1: From the following schema it is clear block(i) cannot receive the sweep j+1 column from block(i+1)
  //        before that block(i) send the column of sweep j to block(i-1).
  //
  //           block (i-1) |       block(i)      |    block (i+1)
  // ...
  // sweep j:   ... -> CS -SW-> CS -> ... -> CS -SW-> CS -> ...
  //                           /                      /
  //                     /-SC--                 /-SC--
  //                    v                      v
  // sweep j+1: ... -> CS -SW-> CS -> ... -> CS -SW-> CS -> ...
  //                           /                      /
  //                     /-SC--                 /-SC--
  //                    v                      v
  // sweep j+2: ... -> CS -SW-> CS -> ... -> CS -SW-> CS -> ...
  // ...
  // where CS is a continue sweep task, SW is a Worker communication and SC is a column communication.
  // Note: Only the dependencies relevant to the analysis are depicted here.
  template <bool dist2 = dist, std::enable_if_t<dist2 && dist == dist2, int> = 0>
  BandBlock(SizeType n, SizeType band_size, SizeType id, SizeType block_size)
      : size_(n), band_size_(band_size), ld_(2 * band_size_ - 1), id_(id), block_size_(block_size),
        mem_size_col_(2 + block_size + (id == 0 ? block_size : 0)), mem_(mem_size_col_ * (ld_ + 1)) {
    using util::ceilDiv;
    DLAF_ASSERT(0 <= n, n);
    // Note: band_size_ = 1 means already tridiagonal.
    DLAF_ASSERT(2 <= band_size, band_size_);
    DLAF_ASSERT(2 <= block_size_, block_size_);
    DLAF_ASSERT(block_size_ % band_size_ == 0, block_size_, band_size_);
    DLAF_ASSERT(0 <= id && id < ceilDiv(size_, block_size_), id, ceilDiv(size_, block_size_));
  }

  T* ptr(SizeType offset, SizeType j) noexcept {
    DLAF_ASSERT_HEAVY(0 <= offset && offset < ld_ + 1, offset, ld_);
    DLAF_ASSERT_HEAVY(0 <= j && j < size_, j, size_);

    if constexpr (dist) {
      return mem_(memory_index(j) * (ld_ + 1) + offset);
    }
    else {
      return mem_(j * (ld_ + 1) + offset);
    }
  }

  SizeType ld() const noexcept {
    return ld_;
  }

  template <Device D, class Sender>
  auto copy_diag(SizeType j, Sender source) noexcept {
    using dlaf::internal::transform;
    namespace ex = pika::execution::experimental;

    constexpr auto B = dlaf::matrix::internal::CopyBackend_v<D, Device::CPU>;

    if constexpr (D == Device::CPU) {
      return transform(
          dlaf::internal::Policy<B>(pika::execution::thread_priority::high),
          [j, this](const matrix::Tile<const T, D>& source) {
            constexpr auto General = blas::Uplo::General;
            constexpr auto Lower = blas::Uplo::Lower;

            common::internal::SingleThreadedBlasScope single;

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
#ifdef DLAF_WITH_GPU
    else if constexpr (D == Device::GPU) {
      DLAF_ASSERT_HEAVY(is_accessible_from_GPU(), "BandBlock memory should be accessible from GPU");
      return transform(
          dlaf::internal::Policy<B>(pika::execution::thread_priority::high),
          [j, this](const matrix::Tile<const T, D>& source, whip::stream_t stream) {
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
  auto copy_off_diag(const SizeType j, Sender source) noexcept {
    using dlaf::internal::transform;

    namespace ex = pika::execution::experimental;

    constexpr auto B = dlaf::matrix::internal::CopyBackend_v<D, Device::CPU>;

    if constexpr (D == Device::CPU) {
      return transform(
          dlaf::internal::Policy<B>(pika::execution::thread_priority::high),
          [j, this](const matrix::Tile<const T, D>& source) {
            constexpr auto General = blas::Uplo::General;
            constexpr auto Upper = blas::Uplo::Upper;

            common::internal::SingleThreadedBlasScope single;

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
#ifdef DLAF_WITH_GPU
    else if constexpr (D == Device::GPU) {
      DLAF_ASSERT_HEAVY(is_accessible_from_GPU(), "BandBlock memory should be accessible from GPU");
      return transform(
          dlaf::internal::Policy<B>(pika::execution::thread_priority::high),
          [j, this](const matrix::Tile<const T, D>& source, whip::stream_t stream) {
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

  SizeType next_split(SizeType j) {
    return mem_size_col_ - memory_index(j);
  }

private:
#ifdef DLAF_WITH_GPU
  bool is_accessible_from_GPU() const {
#ifdef DLAF_WITH_CUDA
    cudaPointerAttributes attrs;
    if (auto status = cudaPointerGetAttributes(&attrs, mem_()); status != cudaSuccess) {
      throw whip::exception(status);
    }
    return cudaMemoryTypeUnregistered != attrs.type;
#elif defined DLAF_WITH_HIP
    // We don't have a similar way to check for accessibility from a device in
    // HIP so we assume that it's always possible. Invalid accesses will result
    // in segmentation faults instead.
    return true;
#endif
  }
#endif

  SizeType memory_index(SizeType j) {
    if constexpr (dist) {
      DLAF_ASSERT_HEAVY(block_size_ * id_ <= j && j < size_, j, id_, block_size_, size_);
      return (j - block_size_ * id_) % mem_size_col_;
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
  SizeType mem_size_col_;

  memory::MemoryView<T, Device::CPU> mem_;
};

template <class CommSender, class T, class DepSender>
[[nodiscard]] auto schedule_send_col(CommSender&& pcomm, comm::IndexT_MPI dest, comm::IndexT_MPI tag,
                                     SizeType b, std::shared_ptr<BandBlock<T, true>> a_block, SizeType j,
                                     DepSender&& dep) {
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::whenAllLift;

  namespace ex = pika::execution::experimental;

  auto send = [dest, tag, b, j](const comm::Communicator& comm,
                                std::shared_ptr<BandBlock<T, true>>& a_block, MPI_Request* req) {
    DLAF_MPI_CHECK_ERROR(MPI_Isend(a_block->ptr(0, j), to_int(2 * b), dlaf::comm::mpi_datatype<T>::type,
                                   dest, tag, comm, req));
  };
  return whenAllLift(std::forward<CommSender>(pcomm), std::move(a_block), std::forward<DepSender>(dep)) |
         transformMPI(send);
}

template <class CommSender, class T, class DepSender>
[[nodiscard]] auto schedule_recv_col(CommSender&& pcomm, comm::IndexT_MPI src, comm::IndexT_MPI tag,
                                     SizeType b, std::shared_ptr<BandBlock<T, true>> a_block, SizeType j,
                                     DepSender&& dep) {
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::whenAllLift;

  auto recv = [src, tag, b, j](const comm::Communicator& comm,
                               std::shared_ptr<BandBlock<T, true>>& a_block, MPI_Request* req) {
    DLAF_MPI_CHECK_ERROR(MPI_Irecv(a_block->ptr(0, j), to_int(2 * b), dlaf::comm::mpi_datatype<T>::type,
                                   src, tag, comm, req));
  };

  return whenAllLift(std::forward<CommSender>(pcomm), std::move(a_block), std::forward<DepSender>(dep)) |
         transformMPI(recv);
}

template <class T>
using BandBlockDist = BandBlock<T, true>;

template <class T>
class SweepWorker {
public:
  SweepWorker(SizeType size, SizeType band_size)
      : size_(size), band_size_(band_size), data_(1 + 2 * band_size) {}

  SweepWorker(const SweepWorker&) = delete;
  SweepWorker(SweepWorker&&) = default;

  SweepWorker& operator=(const SweepWorker&) = delete;
  SweepWorker& operator=(SweepWorker&&) = default;

  void start_sweep(SizeType sweep, BandBlock<T>& a) noexcept {
    start_sweep_internal(sweep, a);
  }

  void compact_copy_to_tile(const matrix::Tile<T, Device::CPU>& tile_v,
                            TileElementIndex index) const noexcept {
    tile_v(index) = tau();
    common::internal::SingleThreadedBlasScope single;
    blas::copy(size_HHR() - 1, v() + 1, 1, tile_v.ptr(index) + 1, 1);
  }

  void do_step(BandBlock<T>& a) noexcept {
    do_step_full(a);
  }

protected:
  template <class BandBlockType>
  void start_sweep_internal(SizeType sweep, BandBlockType& a) noexcept {
    SizeType n = std::min(size_ - sweep - 1, band_size_);
    HH_reflector(n, tau(), v(), a.ptr(1, sweep));

    set_id(sweep, 0);
  }

  template <class BandBlockType>
  void do_step_full(BandBlockType& a) noexcept {
    SizeType j = first_row_HHR();
    SizeType n = size_HHR();  // size diagonal tile and width off-diag tile
    SizeType m = std::min(band_size_, size_ - band_size_ - j);  // height off diagonal tile

    apply_HH_left_right_herm(n, tau(), v(), a.ptr(0, j), a.ld(), w());
    if (m > 0) {
      apply_HH_right(m, n, tau(), v(), a.ptr(n, j), a.ld(), w());
    }
    if (m > 1) {
      HH_reflector(m, tau(), v(), a.ptr(n, j));
      apply_HH_left(m, n - 1, tau(), v(), a.ptr(n - 1, j + 1), a.ld(), w());
    }
    step_ += 1;
    // Note: the sweep is completed if m <= 1.
  }

  void set_id(SizeType sweep, SizeType step) noexcept {
    sweep_ = sweep;
    step_ = step;
  }

  SizeType first_row_HHR() const noexcept {
    return 1 + sweep_ + step_ * band_size_;
  }

  SizeType size_HHR() const noexcept {
    return std::min(band_size_, size_ - first_row_HHR());
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

template <class T>
class SweepWorkerDist : private SweepWorker<T> {
public:
  SweepWorkerDist(SizeType size, SizeType band_size) : SweepWorker<T>(size, band_size) {}

  void start_sweep(SizeType sweep, BandBlockDist<T>& a) {
    this->start_sweep_internal(sweep, a);
  }

  void do_step(BandBlockDist<T>& a) noexcept {
    SizeType j = this->first_row_HHR();
    SizeType n = this->size_HHR();  // size diagonal tile and width off-diag tile

    const auto n1 = a.next_split(j);
    if (n1 < n) {
      do_step_split(a, n1);
    }
    else {
      this->do_step_full(a);
    }
  }

  void send(const comm::Communicator& comm, comm::IndexT_MPI dest, comm::IndexT_MPI tag,
            MPI_Request* req) const noexcept {
    DLAF_MPI_CHECK_ERROR(MPI_Isend(data_(), to_int(band_size_ + 1), comm::mpi_datatype<T>::type, dest,
                                   tag, comm, req));
  }

  void recv(SizeType sweep, SizeType step, const comm::Communicator& comm, comm::IndexT_MPI src,
            comm::IndexT_MPI tag, MPI_Request* req) noexcept {
    SweepWorker<T>::set_id(sweep, step);
    DLAF_MPI_CHECK_ERROR(MPI_Irecv(data_(), to_int(band_size_ + 1), comm::mpi_datatype<T>::type, src,
                                   tag, comm, req));
  }

  using SweepWorker<T>::compact_copy_to_tile;

private:
  void do_step_split(BandBlockDist<T>& a, SizeType n1) noexcept {
    SizeType j = this->first_row_HHR();
    SizeType n = this->size_HHR();  // size diagonal tile and width off-diag tile
    SizeType m = std::min(band_size_, size_ - band_size_ - j);  // height off diagonal tile
    const auto n2 = n - n1;

    apply_HH_left_right_herm(n1, n2, tau(), v(), a.ptr(0, j), a.ptr(0, j + n1), a.ld(), w());
    if (m > 0) {
      apply_HH_right(m, n1, n2, tau(), v(), a.ptr(n, j), a.ptr(n2, j + n1), a.ld(), w());
    }
    if (m > 1) {
      HH_reflector(m, tau(), v(), a.ptr(n, j));
      apply_HH_left(m, n1 - 1, n2, tau(), v(), a.ptr(n - 1, j + 1), a.ptr(n2, j + n1), a.ld(), w());
    }
    step_ += 1;
    // Note: the sweep is completed if m <= 1.
  }

  using SweepWorker<T>::size_;
  using SweepWorker<T>::band_size_;
  using SweepWorker<T>::sweep_;
  using SweepWorker<T>::step_;
  using SweepWorker<T>::data_;
  using SweepWorker<T>::tau;
  using SweepWorker<T>::v;
  using SweepWorker<T>::w;
};

template <class CommSender, class PromiseSender>
[[nodiscard]] auto schedule_send_worker(CommSender&& pcomm, comm::IndexT_MPI dest, comm::IndexT_MPI tag,
                                        PromiseSender&& worker) {
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::whenAllLift;

  auto send = [dest, tag](const comm::Communicator& comm, const auto& worker, MPI_Request* req) {
    worker.send(comm, dest, tag, req);
  };

  return whenAllLift(std::forward<CommSender>(pcomm), std::forward<PromiseSender>(worker)) |
         transformMPI(send);
}

template <class CommSender, class PromiseSender>
[[nodiscard]] auto schedule_recv_worker(SizeType sweep, SizeType step, CommSender&& pcomm,
                                        comm::IndexT_MPI src, comm::IndexT_MPI tag,
                                        PromiseSender&& worker) {
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::whenAllLift;

  auto recv = [sweep, step, src, tag](const comm::Communicator& comm, auto& worker, MPI_Request* req) {
    worker.recv(sweep, step, comm, src, tag, req);
  };

  return whenAllLift(std::forward<CommSender>(pcomm), std::forward<PromiseSender>(worker)) |
         transformMPI(recv);
}

template <Device D, class T>
TridiagResult<T, Device::CPU> BandToTridiag<Backend::MC, D, T>::call_L(
    const SizeType b, Matrix<const T, D>& mat_a) noexcept {
  // Note on the algorithm and dependency tracking:
  // The algorithm is composed by n-2 (real) or n-1 (complex) sweeps:
  // The i-th sweep is initialized by init_sweep/init_sweep_copy_tridiag
  // which act on the i-th column of the band matrix (copy tridiag on previous nb colums as well).
  // Then the sweep is continued using run_sweep.
  // Dependencies are tracked with counting semaphores.
  // In the next schema above each sweep the acquisitions of the semaphore are depicted with an a.
  // Below the sweep the releases are denoted with r.
  //
  // assuming b = 4 and nb = 8:
  // Copy of band: A A A A B B B B C C C C D D D D E E E E F F F
  //                            2r|             2r|         2r+r|
  //               a|a      |a      |a      |a      |a      |a  |
  // Sweep 0       I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5
  //                |      r|      r|      r|      r|      r|r+r|
  //                 a|a      |a      |a      |a      |a      |
  // Sweep 1         I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 *
  //                  |      r|      r|      r|      r|    r+r|
  //                   a|a      |a      |a      |a      |a      |
  // Sweep 2           I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
  //                    |      r|      r|      r|      r|    r+r|
  //                     a|a      |a      |a      |a      |a    |
  // Sweep 3             I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4
  //                      |      r|      r|      r|      r|  r+r|
  //                    ...
  // Note: The last step has an extra release (+r) to ensure that the last step of the next sweep
  //       can execute. E.g. see sweep 3 step 4.

  using common::Pipeline;
  using common::internal::vector;
  using util::ceilDiv;

  using pika::resource::get_num_threads;
  using SemaphorePtr = std::shared_ptr<pika::counting_semaphore<>>;
  using TileVector = std::vector<matrix::Tile<T, Device::CPU>>;
  using TileVectorPtr = std::shared_ptr<TileVector>;

  namespace ex = pika::execution::experimental;

  const auto policy_hp = dlaf::internal::Policy<Backend::MC>(pika::execution::thread_priority::high);

  // note: A is square and has square blocksize
  const SizeType size = mat_a.size().cols();
  const SizeType n = mat_a.nrTiles().cols();
  const SizeType nb = mat_a.blockSize().cols();

  // Need share pointer to keep the allocation until all the tasks are executed.
  auto a_ws = std::make_shared<BandBlock<T>>(size, b);

  Matrix<BaseType<T>, Device::CPU> mat_trid({size, 2}, {nb, 2});
  Matrix<T, Device::CPU> mat_v({size, size}, {nb, nb});

  if (size == 0) {
    return {std::move(mat_trid), std::move(mat_v)};
  }

  auto copy_diag = [a_ws](SizeType j, auto source) {
    return a_ws->template copy_diag<D>(j, std::move(source));
  };

  auto copy_offdiag = [a_ws](SizeType j, auto source) {
    return a_ws->template copy_off_diag<D>(j, std::move(source));
  };

  auto sem = std::make_shared<pika::counting_semaphore<>>(0);

  ex::unique_any_sender<> prev_dep = ex::just();
  // Copy the band matrix
  for (SizeType k = 0; k < n; ++k) {
    SizeType nr_releases = nb / b;
    ex::unique_any_sender<> dep = copy_diag(k * nb, mat_a.read(GlobalTileIndex{k, k}));
    if (k < n - 1) {
      dep = copy_offdiag(k * nb, ex::when_all(std::move(dep), mat_a.read(GlobalTileIndex{k + 1, k})));
    }
    else {
      // Add one to make sure to unlock the last step of the first sweep.
      nr_releases = ceilDiv(size - k * nb, b) + 1;
    }
    prev_dep = ex::when_all(ex::just(nr_releases, sem), std::move(prev_dep), std::move(dep)) |
               ex::then([](SizeType nr, auto&& sem) { sem->release(nr); });
  }
  ex::start_detached(std::move(prev_dep));

  // Maximum size / (2b-1) sweeps can be executed in parallel.
  const auto max_workers =
      std::min(ceilDiv(size, 2 * b - 1), 2 * to_SizeType(get_num_threads("default")));

  vector<Pipeline<SweepWorker<T>>> workers;
  workers.reserve(max_workers);
  for (SizeType i = 0; i < max_workers; ++i)
    workers.emplace_back(SweepWorker<T>(size, b));

  auto run_sweep = [a_ws, size, nb, b](SemaphorePtr&& sem, SemaphorePtr&& sem_next, SizeType sweep,
                                       SweepWorker<T>& worker, const TileVectorPtr& tiles_v) {
    const SizeType nr_steps = nrStepsForSweep(sweep, size, b);
    for (SizeType step = 0; step < nr_steps; ++step) {
      SizeType j_el_tl = sweep % nb;
      // i_el is the row element index with origin in the first row of the diagonal tile.
      SizeType i_el = j_el_tl / b * b + step * b;
      worker.compact_copy_to_tile((*tiles_v)[to_sizet(i_el / nb)], TileElementIndex(i_el % nb, j_el_tl));
      sem->acquire();
      worker.do_step(*a_ws);
      sem_next->release(1);
    }
    // Make sure to unlock the last step of the next sweep
    sem_next->release(1);
  };

  auto copy_tridiag_task = [a_ws](SizeType start, SizeType n_d, SizeType n_e,
                                  const matrix::Tile<BaseType<T>, Device::CPU>& tile_t) {
    DLAF_ASSERT_HEAVY(n_e >= 0 && (n_e == n_d || n_e == n_d - 1), n_e, n_d);
    DLAF_ASSERT_HEAVY(tile_t.size().cols() == 2, tile_t);
    DLAF_ASSERT_HEAVY(tile_t.size().rows() >= n_d, tile_t, n_d);

    auto inc = a_ws->ld() + 1;
    if (isComplex_v<T>)
      // skip imaginary part if Complex.
      inc *= 2;

    common::internal::SingleThreadedBlasScope single;
    blas::copy(n_d, (BaseType<T>*) a_ws->ptr(0, start), inc, tile_t.ptr({0, 0}), 1);
    blas::copy(n_e, (BaseType<T>*) a_ws->ptr(1, start), inc, tile_t.ptr({0, 1}), 1);
  };

  auto init_sweep = [a_ws](SemaphorePtr&& sem, SizeType sweep, SweepWorker<T>& worker) {
    sem->acquire();
    worker.start_sweep(sweep, *a_ws);
    return std::move(sem);
  };

  auto init_sweep_copy_tridiag = [a_ws, copy_tridiag_task,
                                  nb](SemaphorePtr&& sem, SizeType sweep, SweepWorker<T>& worker,
                                      const matrix::Tile<BaseType<T>, Device::CPU>& tile_t) {
    sem->acquire();
    worker.start_sweep(sweep, *a_ws);
    copy_tridiag_task(sweep - (nb - 1), nb, nb, tile_t);
    return std::move(sem);
  };

  const SizeType sweeps = nrSweeps<T>(size);
  ex::any_sender<TileVectorPtr> tiles_v;

  for (SizeType sweep = 0; sweep < sweeps; ++sweep) {
    auto& w_pipeline = workers[sweep % max_workers];
    auto sem_next = std::make_shared<pika::counting_semaphore<>>(0);
    ex::unique_any_sender<SemaphorePtr> sem_sender;
    if ((sweep + 1) % nb != 0) {
      sem_sender = ex::ensure_started(ex::when_all(ex::just(std::move(sem), sweep), w_pipeline()) |
                                      dlaf::internal::transform(policy_hp, init_sweep));
    }
    else {
      const auto tile_index = sweep / nb;
      sem_sender = ex::ensure_started(ex::when_all(ex::just(std::move(sem), sweep), w_pipeline(),
                                                   mat_trid.readwrite(GlobalTileIndex{tile_index, 0})) |
                                      dlaf::internal::transform(policy_hp, init_sweep_copy_tridiag));
    }
    if (sweep % nb == 0) {
      // The run_sweep tasks writes a single column of elements of mat_v.
      // To avoid to retile the matrix (to avoid to have too many tiles), each column of tiles should
      // be shared in read/write mode by multiple tasks.
      // Therefore we extract the tiles of the column in a vector and move it to a shared_ptr,
      // that can be copied to the different tasks, but reference the same tiles.
      const SizeType i = sweep / nb;
      tiles_v =
          ex::when_all_vector(matrix::select(mat_v, common::iterate_range2d(LocalTileIndex{i, i},
                                                                            LocalTileSize{n - i, 1}))) |
          ex::then([](TileVector&& vector) { return std::make_shared<TileVector>(std::move(vector)); }) |
          ex::split();
    }

    ex::when_all(std::move(sem_sender), ex::just(sem_next, sweep), w_pipeline(), tiles_v) |
        dlaf::internal::transformDetach(policy_hp, run_sweep);
    sem = std::move(sem_next);
  }

  auto copy_tridiag = [policy_hp, a_ws, size, nb, &mat_trid, copy_tridiag_task](SizeType i, auto&& dep) {
    const auto tile_index = (i - 1) / nb;
    const auto start = tile_index * nb;
    ex::when_all(ex::just(start, std::min(nb, size - start), std::min(nb, size - 1 - start)),
                 mat_trid.readwrite(GlobalTileIndex{tile_index, 0}), std::forward<decltype(dep)>(dep)) |
        dlaf::internal::transformDetach(policy_hp, copy_tridiag_task);
  };

  auto dep = ex::just(std::move(sem)) |
             dlaf::internal::transform(policy_hp, [](SemaphorePtr&& sem) { sem->acquire(); }) |
             ex::split();

  // copy the last elements of the diagonals
  // As for real types only size - 2 sweeps are performed, make sure that all the elements are copied.
  if (!isComplex_v<T> && (size - 1) % nb == 0) {
    copy_tridiag(size - 1, dep);
  }
  copy_tridiag(size, std::move(dep));

  return {std::move(mat_trid), std::move(mat_v)};
}

struct VAccessHelper {
  // As panel_v are populated in the following way (e.g. with 4 ranks, nb = 2b and tiles_per_block = 2):
  //   Rank 0  Rank1  Rank2  Rank3     Rank 0  Rank1  Rank2  Rank3
  //   A0      A4      A8      A12     B0      B4      B8      B12
  //   A1      A5      A9      A13     B1      B5      B9      B13
  //   --      --      ---     ---     --      --      ---     ---
  //   A2      A6      A10     A14     B2      B6      B10     B14
  //   A3      A7      A11     A15     B3      B7      B11     B15
  //   ---     ---     ---     ---     ---     ---     ---     ---
  //   A16     A20     A24     A28     B16     B20     B24     B28
  //   ...     ...     ...     ...     ...     ...     ...     ...
  //
  // (XN represent a compact (b x b) HH reflector block)
  //
  // and mat_v is populated in the following way:
  //   A0 ** | ** ** | ** ..
  //   A1 B0 | ** ** | ** ..
  //   -- --   -- --   --
  //   A2 B1 | C0 ** | ** ..
  //   A3 B2 | C1 D0 | ** ..
  //   -- --   -- --   --
  //   A4 B3 | C2 D1 | E0 ..
  //   .. .. | .. .. | .. ..
  //
  // the communication of a tile of a panel might be splitted in two parts (e.g. B2 B3 tile).
  // If copyIsSplitted() is true the copy/communication has to happen in two parts Top and Bottom,
  // otherwise only Top is set.

  VAccessHelper(const comm::CommunicatorGrid& grid, const SizeType band, const SizeType sweeps,
                const SizeType sweep0, const SizeType step0, const matrix::Distribution& dist_band,
                const matrix::Distribution& dist_panel, const matrix::Distribution& dist_v) noexcept {
    rank_panel_ = rankPanel(band, step0, dist_band);
    const auto rank = grid.rankFullCommunicator(grid.rank());
    if (rank == rank_panel_)
      index_panel_ = indexPanel(band, step0, dist_band, dist_panel);

    const GlobalElementIndex id_v{(sweep0 / band + step0) * band, sweep0};
    index_v_ = dist_v.globalTileIndex(id_v);
    index_element_tile_v_ = dist_v.tileElementIndex(id_v);

    rank_v_top_ = grid.rankFullCommunicator(dist_v.rankGlobalTile(index_v_));

    const SizeType rows_panel =
        std::min(dist_panel.blockSize().rows(), dist_v.size().rows() - 1 - id_v.row());
    const SizeType rows_v_top = dist_v.tileSize(index_v_).rows() - index_element_tile_v_.row();

    const SizeType cols = std::min(rows_panel, std::min(band, sweeps - sweep0));

    if (rows_v_top < rows_panel) {
      rank_v_bottom_ = grid.rankFullCommunicator(dist_v.rankGlobalTile(indexVBottomInternal(index_v_)));
      size_top_ = TileElementSize{rows_v_top, cols};
      size_bottom_ = TileElementSize{rows_panel - rows_v_top, cols};
    }
    else {
      rank_v_bottom_ = -1;
      size_top_ = TileElementSize{rows_panel, cols};
      size_bottom_ = TileElementSize{0, 0};
    }
  }

  bool copyIsSplitted() const noexcept {
    return size_bottom_.rows() > 0;
  }

  LocalTileIndex indexPanel() const noexcept {
    DLAF_ASSERT_HEAVY(index_panel_.isValid(), index_panel_);
    return index_panel_;
  }
  comm::IndexT_MPI rankPanel() const noexcept {
    return rank_panel_;
  }

  matrix::SubTileSpec specPanelTop() const noexcept {
    return {{0, 0}, size_top_};
  }
  matrix::SubTileSpec specPanelBottom() const noexcept {
    DLAF_ASSERT_HEAVY(copyIsSplitted(), size_bottom_);
    return {{size_top_.rows(), 0}, size_bottom_};
  }

  GlobalTileIndex indexVTop() const noexcept {
    return index_v_;
  }
  matrix::SubTileSpec specVTop() const noexcept {
    return {index_element_tile_v_, size_top_};
  }
  comm::IndexT_MPI rankVTop() const noexcept {
    return rank_v_top_;
  }
  GlobalTileIndex indexVBottom() const noexcept {
    DLAF_ASSERT_HEAVY(copyIsSplitted(), size_bottom_);
    return indexVBottomInternal(index_v_);
  }
  matrix::SubTileSpec specVBottom() const noexcept {
    DLAF_ASSERT_HEAVY(copyIsSplitted(), size_bottom_);
    return {{0, index_element_tile_v_.col()}, size_bottom_};
  }
  comm::IndexT_MPI rankVBottom() const noexcept {
    DLAF_ASSERT_HEAVY(copyIsSplitted(), size_bottom_);
    return rank_v_bottom_;
  }

  static comm::IndexT_MPI rankPanel(const SizeType band, const SizeType step,
                                    const matrix::Distribution& dist_band) noexcept {
    // Need to use dist_band to identify the rank
    // as the panel distribution is local (mismatch between tile size and distribution block-size).
    const GlobalElementIndex id{0, step * band};
    const GlobalTileIndex index = dist_band.globalTileIndex(id);

    return dist_band.rankGlobalTile(index).col();
  }

  static LocalTileIndex indexPanel(const SizeType band, const SizeType step,
                                   const matrix::Distribution& dist_band,
                                   const matrix::Distribution& dist_panel) noexcept {
    // Need to use dist_band to compute the local element index
    // as dist_panel is local (mismatch between tile size and distribution block-size).
    // Then dist_panel is used to compute the local tile index.
    const GlobalElementIndex id{0, step * band};
    const GlobalTileIndex index = dist_band.globalTileIndex(id);

    DLAF_ASSERT_HEAVY(dist_band.rankIndex() == dist_band.rankGlobalTile(index), dist_band.rankIndex(),
                      dist_band.rankGlobalTile(index));

    const SizeType local_row_panel_v =
        dist_band.localTileIndex(index).col() * dist_band.blockSize().cols() +
        dist_band.tileElementIndex(id).col();

    DLAF_ASSERT_HEAVY(dist_panel.tileElementIndex(GlobalElementIndex{local_row_panel_v, 0}) ==
                          TileElementIndex(0, 0),
                      local_row_panel_v, dist_panel.blockSize().rows());

    return dist_panel.localTileIndex(dist_panel.globalTileIndex(GlobalElementIndex{local_row_panel_v,
                                                                                   0}));
  }

private:
  static GlobalTileIndex indexVBottomInternal(const GlobalTileIndex& index_v_top) noexcept {
    return index_v_top + GlobalTileSize{1, 0};
  }

  LocalTileIndex index_panel_;
  comm::IndexT_MPI rank_panel_;
  GlobalTileIndex index_v_;
  TileElementIndex index_element_tile_v_;
  comm::IndexT_MPI rank_v_top_;
  comm::IndexT_MPI rank_v_bottom_;
  TileElementSize size_top_{0, 0};
  TileElementSize size_bottom_{0, 0};
};

template <Device D, class T>
TridiagResult<T, Device::CPU> BandToTridiag<Backend::MC, D, T>::call_L(
    comm::CommunicatorGrid grid, const SizeType b, Matrix<const T, D>& mat_a) noexcept {
  // Note on the algorithm, data distribution and dependency tracking:
  // The band matrix is redistribuited in 1D block cyclic. The new block size is a multiple of the
  // block_size of mat_a. As sweeps are performed the matrix is shifted one column to the left (The
  // computed diagonal and off diagonal elements of the resulting tridiagonal matrix are copied into the
  // result such that the column of the buffer can be reused.)
  //
  // The algorithm is composed by n-2 (real) or n-1 (complex) sweeps:
  // The i-th sweep is initialized by init_sweep which act on the i-th column of the band matrix.
  // Then the sweep continues applying steps.
  // The j-th step acts on the columns [i+1 + j * b, i+1 + (j+1) * b)
  // The steps in the same sweep has to be executed in order and the dependencies are managed by the
  // worker pipelines. The deps vector is used to set the dependencies among two different sweeps.
  //
  // assuming b = 4 and nb = 8 (i.e each task applies two steps) distributed with block 2 * nb (i.e.
  // tiles_per_block = 2):
  //               RANK 0                                                 RANK 1
  // Copy of band: A A A A B B B B C C C C D D D D                        E E E E F F F F G ...
  //                 deps[0][0]   |  deps[0][1]   | deps[0][2]      <-col-  deps[0][0]   |  ...
  // Sweep 0       I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3        -worker->       4 4 4 4 5 5 5 5
  //                |  deps[0][0]   |  deps[0][1]   | deps[0][2]    <-col-    deps[0][0]   |  ...
  // Sweep 1         I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3      -worker->         4 4 4 4 5 5 5 5 ...
  //                  |  deps[0][0]   |  deps[0][1]   | deps[0][2]  <-col-      deps[0][0]   |  ...
  // Sweep 2           I 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3    -worker->           4 4 4 4 5 5 5 5 ...
  //                    ...
  // Note: j-th task (in this case 2*j-th and 2*j+1-th steps) depends explicitly only on deps[*][j+1],
  //       as the pipeline dependency on j-1-th task (or sweep_init for j=0) implies a dependency on
  //       deps[*][j] as well.
  using common::iterate_range2d;
  using common::Pipeline;
  using common::RoundRobin;
  using common::internal::vector;
  using dlaf::internal::Policy;
  using matrix::copy;
  using matrix::internal::CopyBackend_v;
  using util::ceilDiv;

  using pika::resource::get_num_threads;

  namespace ex = pika::execution::experimental;

  // Should be dispatched to local implementation if (1x1) grid.
  DLAF_ASSERT(grid.size() != comm::Size2D(1, 1), grid);

  // note: A is square and has square blocksize
  SizeType size = mat_a.size().cols();
  SizeType nrtile = mat_a.nrTiles().cols();
  SizeType nb = mat_a.blockSize().cols();
  auto& dist_a = mat_a.distribution();

  Matrix<BaseType<T>, Device::CPU> mat_trid({size, 2}, {nb, 2});
  matrix::Distribution dist_v({size, size}, {nb, nb}, dist_a.commGridSize(), dist_a.rankIndex(),
                              dist_a.sourceRankIndex());
  Matrix<T, Device::CPU> mat_v(dist_v);

  if (size == 0) {
    return {std::move(mat_trid), std::move(mat_v)};
  }

  auto comm = ex::just(grid.fullCommunicator().clone());
  // Need a pipeline of comm for broadcasts.
  common::Pipeline<comm::Communicator> comm_bcast(grid.fullCommunicator().clone());

  const auto rank = grid.rankFullCommunicator(grid.rank());
  const auto ranks = static_cast<comm::IndexT_MPI>(grid.size().linear_size());
  const auto prev_rank = (rank == 0 ? ranks - 1 : rank - 1);
  const auto next_rank = (rank + 1 == ranks ? 0 : rank + 1);

  const SizeType nb_band = get1DBlockSize(nb);
  const SizeType tiles_per_block = nb_band / nb;
  matrix::Distribution dist({1, size}, {1, nb_band}, {1, ranks}, {0, rank}, {0, 0});

  // Maximum block_size / (2b-1) sweeps per block can be executed in parallel + 1 communication buffer.
  const auto workers_per_block = 1 + ceilDiv(dist.blockSize().cols(), 2 * b - 1);

  // Point to point communication happens in four ways:
  // - when copying the band matrix in compact form             -> compute_copy_tag
  // - when sending the Worker to the next rank                 -> compute_worker_tag
  // - when sending a column of the matrix to the previous rank -> compute_col_tag
  // - when copying the HH reflectors to v                      -> compute_v_tag
  // and to avoid communication mixing of the different phases tags have to be different.
  // Note: when ranks > 2 compute_worker_tag and compute_col_tag can have the same value
  //       as one communication flows to the left and one to the right.
  auto compute_copy_tag = [](SizeType j, bool is_offdiag) {
    return static_cast<comm::IndexT_MPI>(2 * j + (is_offdiag ? 1 : 0));
  };
  // The offset is set to the first unused tag by compute_copy_tag.
  const comm::IndexT_MPI offset_v_tag = compute_copy_tag(nrtile, false);

  auto compute_v_tag = [offset_v_tag](SizeType i, bool is_bottom) {
    // only the row index is needed as dependencies are added to avoid
    // more columns of the same row at the same time.
    return offset_v_tag + static_cast<comm::IndexT_MPI>(2 * i) + (is_bottom ? 1 : 0);
  };

  // The offset is set to the first unused tag by compute_v_tag.
  const comm::IndexT_MPI offset_col_tag = compute_v_tag(nrtile, false);

  auto compute_col_tag = [offset_col_tag, ranks](SizeType block_id, bool last_col) {
    // By construction the communication from block j+1 to block j are dependent, therefore a tag per
    // block is enough. Moreover block_id is divided by the number of ranks as only the local index is
    // needed.
    // When the last column ((size-1)-th column) is communicated the tag is incremented by 1 as in
    // some case it can mix with the (size-2)-th columnn.
    // Note: Passing the local_block_id is not an option as the sender local index might be different
    //       from the receiver index.
    return offset_col_tag + static_cast<comm::IndexT_MPI>(block_id / ranks) + (last_col ? 1 : 0);
  };

  // Same offset if ranks > 2, otherwise add the first unused tag of compute_col_tag.
  const comm::IndexT_MPI offset_worker_tag =
      offset_col_tag + (ranks == 2 ? compute_col_tag(dist.nrTiles().cols() - 1, true) + 1 : 0);
  ;

  auto compute_worker_tag = [offset_worker_tag, workers_per_block, ranks](SizeType sweep,
                                                                          SizeType block_id) {
    // As only workers_per_block are available a dependency is introduced by reusing it, therefore
    // a different tag for all sweeps is not needed.
    // Moreover block_id is divided by the number of ranks as only the local index is needed.
    // Note: Passing the local_block_id is not an option as the sender local index might be different
    //       from the receiver index.
    return offset_worker_tag + static_cast<comm::IndexT_MPI>(sweep % workers_per_block +
                                                             block_id / ranks * workers_per_block);
  };

  // Need shared pointer to keep the allocation until all the tasks are executed.
  vector<std::shared_ptr<BandBlock<T, true>>> a_ws;
  for (SizeType i = 0; i < dist.localNrTiles().cols(); ++i) {
    a_ws.emplace_back(std::make_shared<BandBlock<T, true>>(size, b, rank + i * ranks, nb_band));
  }

  vector<vector<ex::any_sender<>>> deps(dist.localNrTiles().cols());
  for (auto& dep : deps) {
    dep.reserve(nb_band / nb);
  }

  {
    constexpr std::size_t n_workspaces = 4;
    RoundRobin<Matrix<T, Device::CPU>> temps(n_workspaces, LocalElementSize{nb, nb},
                                             TileElementSize{nb, nb});

    auto copy_diag = [](std::shared_ptr<BandBlock<T, true>> a_block, SizeType j, auto source) {
      constexpr Device device = dlaf::internal::sender_device<decltype(source)>;
      return a_block->template copy_diag<device>(j, std::move(source));
    };

    auto copy_offdiag = [](std::shared_ptr<BandBlock<T, true>> a_block, SizeType j, auto source) {
      constexpr Device device = dlaf::internal::sender_device<decltype(source)>;
      return a_block->template copy_off_diag<device>(j, std::move(source));
    };

    // Copy the band matrix
    for (SizeType k = 0; k < nrtile; ++k) {
      const auto id_block = k / tiles_per_block;
      const GlobalTileIndex index_diag{k, k};
      const GlobalTileIndex index_offdiag{k + 1, k};
      const auto rank_block = dist.rankGlobalTile<Coord::Col>(id_block);
      const auto rank_diag = grid.rankFullCommunicator(dist_a.rankGlobalTile(index_diag));
      const auto rank_offdiag =
          (k == nrtile - 1 ? -1 : grid.rankFullCommunicator(dist_a.rankGlobalTile(index_offdiag)));
      const auto tag_diag = compute_copy_tag(k, false);
      const auto tag_offdiag = compute_copy_tag(k, true);

      if (rank == rank_block) {
        ex::any_sender<> dep;
        const auto id_block_local = dist.localTileFromGlobalTile<Coord::Col>(id_block);

        if (rank == rank_diag) {
          dep = copy_diag(a_ws[id_block_local], k * nb, mat_a.read(index_diag)) | ex::split();
        }
        else {
          auto& temp = temps.nextResource();
          auto diag_tile = comm::scheduleRecv(ex::make_unique_any_sender(comm), rank_diag, tag_diag,
                                              splitTile(temp.readwrite(LocalTileIndex{0, 0}),
                                                        {{0, 0}, dist_a.tileSize(index_diag)}));
          dep = copy_diag(a_ws[id_block_local], k * nb, std::move(diag_tile)) | ex::split();
        }

        if (k < nrtile - 1) {
          if (rank == rank_offdiag) {
            dep = copy_offdiag(a_ws[id_block_local], k * nb,
                               ex::when_all(std::move(dep), mat_a.read(index_offdiag))) |
                  ex::split();
          }
          else {
            auto& temp = temps.nextResource();
            auto offdiag_tile =
                comm::scheduleRecv(ex::make_unique_any_sender(comm), rank_offdiag, tag_offdiag,
                                   splitTile(temp.readwrite(LocalTileIndex{0, 0}),
                                             {{0, 0}, dist_a.tileSize(index_offdiag)}));
            dep = copy_offdiag(a_ws[id_block_local], k * nb,
                               ex::when_all(std::move(dep), std::move(offdiag_tile))) |
                  ex::split();
          }
        }

        deps[id_block_local].push_back(std::move(dep));
      }
      else {
        if (rank == rank_diag) {
          ex::start_detached(comm::scheduleSend(ex::make_unique_any_sender(comm), rank_block, tag_diag,
                                                mat_a.read(index_diag)));
        }
        if (k < nrtile - 1) {
          if (rank == rank_offdiag) {
            ex::start_detached(comm::scheduleSend(ex::make_unique_any_sender(comm), rank_block,
                                                  tag_offdiag, mat_a.read(index_offdiag)));
          }
        }
      }
    }
  }

  vector<vector<Pipeline<SweepWorkerDist<T>>>> workers(dist.localNrTiles().cols());
  for (auto& workers_block : workers) {
    workers_block.reserve(workers_per_block);
    for (SizeType i = 0; i < workers_per_block; ++i)
      workers_block.emplace_back(SweepWorkerDist<T>(size, b));
  }

  constexpr std::size_t n_workspaces = 2;
  // As the panel has tiles of size (nb x b), while it should be distributed with a row block-size
  // of nb * tiles_per_block, we use a local distribution and we manage the computation of the
  // local panel index with VAccessHelper.
  matrix::Distribution dist_panel({dist.localSize().cols(), b}, {nb, b});
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> v_panels(n_workspaces, dist_panel);

  auto init_sweep = [](std::shared_ptr<BandBlock<T, true>> a_block, SizeType sweep,
                       SweepWorkerDist<T>& worker) { worker.start_sweep(sweep, *a_block); };
  auto cont_sweep = [b](std::shared_ptr<BandBlock<T, true>> a_block, SizeType nr_steps,
                        SweepWorkerDist<T>& worker, matrix::Tile<T, Device::CPU>&& tile_v,
                        TileElementIndex index) {
    for (SizeType j = 0; j < nr_steps; ++j) {
      worker.compact_copy_to_tile(tile_v, index + TileElementSize(j * b, 0));
      worker.do_step(*a_block);
    }
  };

  auto policy_hp = dlaf::internal::Policy<Backend::MC>(pika::execution::thread_priority::high);
  auto copy_tridiag = [policy_hp, &mat_trid](std::shared_ptr<BandBlock<T, true>> a_block, SizeType sweep,
                                             auto&& dep) {
    auto copy_tridiag_task = [](std::shared_ptr<BandBlock<T, true>> a_block, SizeType start,
                                SizeType n_d, SizeType n_e, auto tile_t) {
      DLAF_ASSERT_HEAVY(n_e >= 0 && (n_e == n_d || n_e == n_d - 1), n_e, n_d);
      DLAF_ASSERT_HEAVY(tile_t.size().cols() == 2, tile_t);
      DLAF_ASSERT_HEAVY(tile_t.size().rows() >= n_d, tile_t, n_d);

      auto inc = a_block->ld() + 1;
      if (isComplex_v<T>)
        // skip imaginary part if Complex.
        inc *= 2;

      common::internal::SingleThreadedBlasScope single;

      if (auto n1 = a_block->next_split(start); n1 < n_d) {
        blas::copy(n1, (BaseType<T>*) a_block->ptr(0, start), inc, tile_t.ptr({0, 0}), 1);
        blas::copy(n_d - n1, (BaseType<T>*) a_block->ptr(0, start + n1), inc, tile_t.ptr({n1, 0}), 1);
        blas::copy(n1, (BaseType<T>*) a_block->ptr(1, start), inc, tile_t.ptr({0, 1}), 1);
        blas::copy(n_e - n1, (BaseType<T>*) a_block->ptr(1, start + n1), inc, tile_t.ptr({n1, 1}), 1);
      }
      else {
        blas::copy(n_d, (BaseType<T>*) a_block->ptr(0, start), inc, tile_t.ptr({0, 0}), 1);
        blas::copy(n_e, (BaseType<T>*) a_block->ptr(1, start), inc, tile_t.ptr({0, 1}), 1);
      }
    };

    const auto size = mat_trid.size().rows();
    const auto nb = mat_trid.blockSize().rows();
    if (sweep % nb == nb - 1 || sweep == size - 1) {
      const auto tile_index = sweep / nb;
      const auto start = tile_index * nb;
      dlaf::internal::whenAllLift(std::move(a_block), start, std::min(nb, size - start),
                                  std::min(nb, size - 1 - start),
                                  mat_trid.readwrite(GlobalTileIndex{tile_index, 0}),
                                  std::forward<decltype(dep)>(dep)) |
          dlaf::internal::transformDetach(policy_hp, copy_tridiag_task);
    }
    else {
      ex::start_detached(std::forward<decltype(dep)>(dep));
    }
  };

  const SizeType steps_per_block = nb_band / b;
  const SizeType steps_per_task = nb / b;
  const SizeType sweeps = nrSweeps<T>(size);

  for (SizeType sweep = 0; sweep < sweeps; ++sweep) {
    const SizeType steps = nrStepsForSweep(sweep, size, b);

    auto& v_panel = sweep % b == 0 ? v_panels.nextResource() : v_panels.currentResource();

    SizeType last_step = 0;
    for (SizeType init_step = 0; init_step < steps; init_step += steps_per_block) {
      const auto block_id = dist.globalTileIndex(GlobalElementIndex{0, init_step * b});
      const auto rank_block = dist.rankGlobalTile(block_id).col();
      const SizeType block_steps = std::min(steps_per_block, steps - init_step);

      if (prev_rank == rank_block) {
        const SizeType next_j = sweep + (init_step + steps_per_block) * b;
        if (next_j < size) {
          const auto block_local_id = dist.localTileIndex(block_id + GlobalTileSize{0, 1}).col();
          auto a_block = a_ws[block_local_id];
          auto& deps_block = deps[block_local_id];
          ex::start_detached(schedule_send_col(comm, prev_rank,
                                               compute_col_tag(block_id.col(), next_j == size - 1), b,
                                               a_block, next_j, deps_block[0]));
        }
      }
      else if (rank == rank_block) {
        const auto block_local_id = dist.localTileIndex(block_id).col();
        auto a_block = a_ws[block_local_id];
        auto& w_pipeline = workers[block_local_id][sweep % workers_per_block];
        auto& deps_block = deps[block_local_id];

        // Sweep initialization
        if (init_step == 0) {
          auto dep = dlaf::internal::whenAllLift(a_block, sweep, w_pipeline(), deps_block[0]) |
                     dlaf::internal::transform(policy_hp, init_sweep);

          copy_tridiag(a_block, sweep, std::move(dep));
        }
        else {
          ex::start_detached(schedule_recv_worker(sweep, init_step, comm, prev_rank,
                                                  compute_worker_tag(sweep, block_id.col()),
                                                  w_pipeline()));
        }

        // Index of the first column (currently) after this block (if exists).
        const SizeType next_j = sweep + (init_step + steps_per_block) * b;
        if (next_j < size) {
          // The dependency on the operation of the previous sweep is real as the Worker cannot be sent
          // before deps_block.back() gets ready, and the Worker is needed in the next rank to update the
          // column before is sent here.
          deps_block.push_back(schedule_recv_col(comm, next_rank,
                                                 compute_col_tag(block_id.col(), next_j == size - 1), b,
                                                 a_block, next_j, deps_block.back()) |
                               ex::split());
        }

        for (SizeType block_step = 0; block_step < block_steps; block_step += steps_per_task) {
          // Last task only applies the remaining steps to the block boundary
          const SizeType nr_steps = std::min(steps_per_task, block_steps - block_step);

          auto dep_index =
              std::min(ceilDiv(block_step + nr_steps, steps_per_task), deps_block.size() - 1);

          const auto local_index_tile_panel_v =
              VAccessHelper::indexPanel(b, init_step + block_step, dist, dist_panel);

          deps_block[ceilDiv(block_step, steps_per_task)] =
              dlaf::internal::whenAllLift(a_block, nr_steps, w_pipeline(),
                                          v_panel.readwrite(local_index_tile_panel_v),
                                          TileElementIndex{0, sweep % b}, deps_block[dep_index]) |
              dlaf::internal::transform(policy_hp, cont_sweep) | ex::split();

          last_step = block_step;
        }

        // Shrink the dependency vector to only include the senders generated by this block in this sweep.
        deps_block.resize(ceilDiv(last_step, steps_per_task) + 1);

        if (init_step + block_steps < steps) {
          ex::start_detached(schedule_send_worker(
              comm, next_rank, compute_worker_tag(sweep, block_id.col() + 1), w_pipeline()));
        }
      }
    }
    // send HH reflector to the correct rank.
    if ((sweep + 1) % b == 0 || sweep + 1 == sweeps) {
      const SizeType base_sweep = sweep / b * b;
      const SizeType base_sweep_steps = nrStepsForSweep(base_sweep, size, b);

      for (SizeType init_step = 0; init_step < base_sweep_steps; init_step += steps_per_block) {
        const SizeType base_sweep_block_steps = std::min(steps_per_block, base_sweep_steps - init_step);

        for (SizeType block_step = 0; block_step < base_sweep_block_steps;
             block_step += steps_per_task) {
          VAccessHelper helper(grid, b, sweeps, base_sweep, init_step + block_step, dist, dist_panel,
                               dist_v);

          if (rank == helper.rankPanel()) {
            auto copy_or_send =
                [&comm, rank, &v_panel, &mat_v,
                 &compute_v_tag](const LocalTileIndex index_panel, const matrix::SubTileSpec spec_panel,
                                 const comm::IndexT_MPI rank_v, const GlobalTileIndex index_v,
                                 const matrix::SubTileSpec spec_v, const bool bottom) {
                  auto tile_v_panel = splitTile(v_panel.read(index_panel), spec_panel);
                  if (rank == rank_v) {
                    auto tile_v = splitTile(mat_v.readwrite(index_v), spec_v);
                    ex::start_detached(ex::when_all(std::move(tile_v_panel), std::move(tile_v)) |
                                       copy(Policy<CopyBackend_v<Device::CPU, Device::CPU>>{}));
                  }
                  else {
                    ex::start_detached(comm::scheduleSend(ex::make_unique_any_sender(comm), rank_v,
                                                          compute_v_tag(index_v.row(), bottom),
                                                          std::move(tile_v_panel)));
                  }
                };

            copy_or_send(helper.indexPanel(), helper.specPanelTop(), helper.rankVTop(),
                         helper.indexVTop(), helper.specVTop(), false);
            if (helper.copyIsSplitted()) {
              copy_or_send(helper.indexPanel(), helper.specPanelBottom(), helper.rankVBottom(),
                           helper.indexVBottom(), helper.specVBottom(), true);
            }
          }
          else {
            auto recv = [&comm, rank, &dist_v, &mat_v,
                         &compute_v_tag](const comm::IndexT_MPI rank_panel,
                                         const comm::IndexT_MPI rank_v, const GlobalTileIndex index_v,
                                         const matrix::SubTileSpec spec_v, const bool bottom) {
              if (rank == rank_v) {
                auto tile_v = splitTile(mat_v.readwrite(index_v), spec_v);
                auto local_index_v = dist_v.localTileIndex(index_v);

                ex::any_sender<> dep;
                if (local_index_v.col() == 0)
                  dep = ex::just();
                else
                  dep = ex::drop_value(mat_v.read(local_index_v - LocalTileSize{0, 1}));

                ex::start_detached(comm::scheduleRecv(
                    ex::make_unique_any_sender(comm), rank_panel, compute_v_tag(index_v.row(), bottom),
                    matrix::ReadWriteTileSender<T, Device::CPU>(ex::when_all(std::move(tile_v),
                                                                             std::move(dep)))));
              }
            };

            recv(helper.rankPanel(), helper.rankVTop(), helper.indexVTop(), helper.specVTop(), false);
            if (helper.copyIsSplitted()) {
              recv(helper.rankPanel(), helper.rankVBottom(), helper.indexVBottom(), helper.specVBottom(),
                   true);
            }
          }
        }
      }
    }
  }

  // Rank 0 (owner of the first band matrix block) copies the last parts of the tridiag matrix.
  if (rank == 0) {
    // copy the last elements of the diagonals
    if constexpr (!isComplex_v<T>) {
      // only needed for real types as they don't perform sweep size-2
      copy_tridiag(a_ws[0], size - 2, deps[0][0]);
    }
    copy_tridiag(a_ws[0], size - 1, std::move(deps[0][0]));
  }

  // only rank0 has mat_trid -> bcast to everyone.
  for (const auto& index : iterate_range2d(mat_trid.nrTiles())) {
    if (rank == 0)
      ex::start_detached(comm::scheduleSendBcast(comm_bcast(), mat_trid.read(index)));
    else
      ex::start_detached(comm::scheduleRecvBcast(comm_bcast(), 0, mat_trid.readwrite(index)));
  }

  return {std::move(mat_trid), std::move(mat_v)};
}
}
