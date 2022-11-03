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

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/eigensolver/band_to_tridiag/api.h"
#include "dlaf/lapack/gpu/lacpy.h"
#include "dlaf/lapack/gpu/laset.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
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

// split versions of the previous operations
template <class T>
void applyHHLeftRightHerm(const SizeType n1, const SizeType n2, const T tau, const T* v, T* a1, T* a2,
                          const SizeType lda, T* w) {
  DLAF_ASSERT_HEAVY(n1 > 0, n1);
  DLAF_ASSERT_HEAVY(n2 > 0, n2);
  const auto n = n1 + n2;

  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto ColMaj = blas::Layout::ColMajor;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;

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
void applyHHLeft(const SizeType m, const SizeType n1, const SizeType n2, const T tau, const T* v, T* a1,
                 T* a2, const SizeType lda, T* w) {
  applyHHLeft(m, n1, tau, v, a1, lda, w);
  applyHHLeft(m, n2, tau, v, a2, lda, w);
}

template <class T>
void applyHHRight(const SizeType m, const SizeType n1, const SizeType n2, const T tau, const T* v, T* a1,
                  T* a2, const SizeType lda, T* w) {
  DLAF_ASSERT_HEAVY(m >= 0, m);
  DLAF_ASSERT_HEAVY(n1 > 0, n1);
  DLAF_ASSERT_HEAVY(n2 > 0, n2);

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
        mem_size_col_(n), mem_(mem_size_col_ * (ld_ + 1)) {}

  // Distributed constructor
  // TODO document size of allocated memory
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

    if (dist) {
      return mem_(memoryIndex(j) * (ld_ + 1) + offset);
    }
    else {
      return mem_(j * (ld_ + 1) + offset);
    }
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
          dlaf::internal::Policy<B>(pika::execution::thread_priority::high),
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
#ifdef DLAF_WITH_GPU
    else if constexpr (D == Device::GPU) {
      DLAF_ASSERT_HEAVY(isAccessibleFromGPU(), "BandBlock memory should be accessible from GPU");
      return transform(
          dlaf::internal::Policy<B>(pika::execution::thread_priority::high),
          [=](const matrix::Tile<const T, D>& source, whip::stream_t stream) {
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
          dlaf::internal::Policy<B>(pika::execution::thread_priority::high),
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
#ifdef DLAF_WITH_GPU
    else if constexpr (D == Device::GPU) {
      DLAF_ASSERT_HEAVY(isAccessibleFromGPU(), "BandBlock memory should be accessible from GPU");
      return transform(
          dlaf::internal::Policy<B>(pika::execution::thread_priority::high),
          [=](const matrix::Tile<const T, D>& source, whip::stream_t stream) {
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

  SizeType nextSplit(SizeType j) {
    return mem_size_col_ - memoryIndex(j);
  }

private:
#ifdef DLAF_WITH_GPU
  bool isAccessibleFromGPU() const {
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

  SizeType memoryIndex(SizeType j) {
    if (dist) {
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
[[nodiscard]] auto scheduleSendCol(CommSender&& pcomm, comm::IndexT_MPI dest, comm::IndexT_MPI tag,
                                   SizeType b, std::shared_ptr<BandBlock<T, true>> a_block, SizeType j,
                                   DepSender&& dep) {
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::whenAllLift;

  namespace ex = pika::execution::experimental;

  auto send = [dest, tag, b, j](const comm::Communicator& comm,
                                std::shared_ptr<BandBlock<T, true>> a_block, MPI_Request* req) {
    DLAF_MPI_CHECK_ERROR(MPI_Isend(a_block->ptr(0, j), to_int(2 * b), dlaf::comm::mpi_datatype<T>::type,
                                   dest, tag, comm, req));
  };

  auto comm_sender =
      whenAllLift(std::forward<CommSender>(pcomm), a_block, std::forward<DepSender>(dep)) |
      transformMPI(send);
  // Ensure lifetime shared pointer.
  return whenAllLift(std::move(a_block), std::move(comm_sender)) | ex::then([](auto) {});
}

template <class CommSender, class T, class DepSender>
[[nodiscard]] auto scheduleRecvCol(CommSender&& pcomm, comm::IndexT_MPI src, comm::IndexT_MPI tag,
                                   SizeType b, std::shared_ptr<BandBlock<T, true>> a_block, SizeType j,
                                   DepSender&& dep) {
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::whenAllLift;

  auto recv = [src, tag, b, j](const comm::Communicator& comm,
                               std::shared_ptr<BandBlock<T, true>> a_block, MPI_Request* req) {
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
    return *data_();
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

  void startSweep(SizeType sweep, BandBlockDist<T>& a) {
    this->startSweepInternal(sweep, a);
  }

  void doStep(BandBlockDist<T>& a) noexcept {
    SizeType j = this->firstRowHHR();
    SizeType n = this->sizeHHR();  // size diagonal tile and width off-diag tile

    const auto n1 = a.nextSplit(j);
    if (n1 < n) {
      doStepSplit(a, n1);
    }
    else {
      this->doStepFull(a);
    }
  }

  void send(const comm::Communicator& comm, comm::IndexT_MPI dest, comm::IndexT_MPI tag,
            MPI_Request* req) const noexcept {
    DLAF_MPI_CHECK_ERROR(MPI_Isend(const_cast<T*>(data_()), to_int(band_size_ + 1),
                                   comm::mpi_datatype<T>::type, dest, tag, comm, req));
  }

  void recv(SizeType sweep, SizeType step, const comm::Communicator& comm, comm::IndexT_MPI src,
            comm::IndexT_MPI tag, MPI_Request* req) noexcept {
    SweepWorker<T>::setId(sweep, step);
    *data_(0) = T{9};
    *data_(1) = T{9};
    DLAF_MPI_CHECK_ERROR(
        MPI_Irecv(data_(), to_int(band_size_ + 1), comm::mpi_datatype<T>::type, src, tag, comm, req));
  }

  using SweepWorker<T>::compactCopyToTile;

private:
  void doStepSplit(BandBlockDist<T>& a, SizeType n1) noexcept {
    SizeType j = this->firstRowHHR();
    SizeType n = this->sizeHHR();  // size diagonal tile and width off-diag tile
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
  using SweepWorker<T>::data_;
  using SweepWorker<T>::tau;
  using SweepWorker<T>::v;
  using SweepWorker<T>::w;
};

template <class CommSender, class PromiseSender>
[[nodiscard]] auto scheduleSendWorker(CommSender&& pcomm, comm::IndexT_MPI dest, comm::IndexT_MPI tag,
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
[[nodiscard]] auto scheduleRecvWorker(SizeType sweep, SizeType step, CommSender&& pcomm,
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

template <class T>
class vector2D {
public:
  vector2D(SizeType nr, SizeType size) : data_(nr * size), ld_(size) {}

  T& operator()(SizeType block, SizeType index) noexcept {
    return data_[id(block, index)];
  }
  const T& operator()(SizeType block, SizeType index) const noexcept {
    return data_[id(block, index)];
  }

private:
  SizeType id(SizeType block, SizeType index) {
    DLAF_ASSERT_HEAVY(index < ld_, index, ld_);
    DLAF_ASSERT_HEAVY(block < data_.size() / ld_, block, data_.size(), ld_);
    return ld_ * block + index;
  }

  common::internal::vector<T> data_;
  SizeType ld_;
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

  auto policy_hp = dlaf::internal::Policy<Backend::MC>(pika::execution::thread_priority::high);
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

std::ostream& operator<<(std::ostream& out, matrix::SubTileSpec spec) {
  out << spec.origin << "+" << spec.size;
  return out;
}

struct VAccessHelper {
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
    const GlobalElementIndex id{0, step * band};
    const GlobalTileIndex index = dist_band.globalTileIndex(id);

    return dist_band.rankGlobalTile(index).col();
  }

  static LocalTileIndex indexPanel(const SizeType band, const SizeType step,
                                   const matrix::Distribution& dist_band,
                                   const matrix::Distribution& dist_panel) noexcept {
    const GlobalElementIndex id{0, step * band};
    const GlobalTileIndex index = dist_band.globalTileIndex(id);

    DLAF_ASSERT_HEAVY(dist_band.rankIndex() == dist_band.rankGlobalTile(index), dist_band.rankIndex(),
                      dist_band.rankGlobalTile(index));

    const SizeType local_row_panel_v =
        dist_band.localTileIndex(index).col() * dist_band.blockSize().cols() +
        dist_band.tileElementIndex(id).col();
    return dist_panel.localTileIndex(
        dist_panel.globalTileIndex(GlobalElementIndex{local_row_panel_v, 0}));
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

// Distributed implementation of bandToTridiag.
template <Device D, class T>
TridiagResult<T, Device::CPU> BandToTridiagDistr<Backend::MC, D, T>::call_L(
    comm::CommunicatorGrid grid, const SizeType b, Matrix<const T, D>& mat_a) noexcept {
  using common::iterate_range2d;
  using common::Pipeline;
  using common::PromiseGuard;
  using common::RoundRobin;
  using common::internal::vector;
  using dlaf::internal::Policy;
  using matrix::copy;
  using matrix::internal::CopyBackend_v;
  using util::ceilDiv;

  using pika::resource::get_num_threads;

  namespace ex = pika::execution::experimental;

  static_assert(D == Device::CPU);

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

  SizeType tiles_per_block = 1;
  matrix::Distribution dist({1, size}, {1, nb * tiles_per_block}, {1, ranks}, {0, rank}, {0, 0});
  SizeType nb_band = dist.blockSize().cols();

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

  auto compute_v_tag = [offset_v_tag](SizeType i) {
    // only the row index is needed as dependencies are added to avoid
    // more columns of the same row at the same time.
    return offset_v_tag + static_cast<comm::IndexT_MPI>(i);
  };

  // The offset is set to the first unused tag by compute_v_tag.
  const comm::IndexT_MPI offset_col_tag = compute_v_tag(nrtile);

  auto compute_col_tag = [offset_col_tag, ranks](SizeType block_id) {
    // By construction the communication from block j+1 to block j are dependent, therefore a tag per
    // block is enough. Moreover block_id is divided by the number of ranks as only the local index is
    // needed.
    // Note: Passing the local_block_id is not an option as the sender local index might be different
    //       from the receiver index.
    return offset_col_tag + static_cast<comm::IndexT_MPI>(block_id / ranks);
  };

  // Same offset if ranks > 2, otherwise add the first unused tag of compute_col_tag.
  const comm::IndexT_MPI offset_worker_tag =
      offset_col_tag + (ranks == 2 ? compute_col_tag(dist.nrTiles().cols() - 1) + 1 : 0);
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
    RoundRobin<Matrix<T, D>> temps(n_workspaces, LocalElementSize{nb, nb}, TileElementSize{nb, nb});

    auto copy_diag = [](std::shared_ptr<BandBlock<T, true>> a_block, SizeType j, auto source) {
      return a_block->template copyDiag<D>(j, std::move(source));
    };

    auto copy_offdiag = [](std::shared_ptr<BandBlock<T, true>> a_block, SizeType j, auto source) {
      return a_block->template copyOffDiag<D>(j, std::move(source));
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
          dep = copy_diag(a_ws[id_block_local], k * nb, mat_a.read_sender(index_diag)) | ex::split();
        }
        else {
          auto& temp = temps.nextResource();
          auto diag_tile = comm::scheduleRecv(comm, rank_diag, tag_diag,
                                              splitTile(temp(LocalTileIndex{0, 0}),
                                                        {{0, 0}, dist_a.tileSize(index_diag)}));
          dep = copy_diag(a_ws[id_block_local], k * nb, std::move(diag_tile)) | ex::split();
        }

        if (k < nrtile - 1) {
          if (rank == rank_offdiag) {
            dep = copy_offdiag(a_ws[id_block_local], k * nb,
                               ex::when_all(std::move(dep), mat_a.read_sender(index_offdiag))) |
                  ex::split();
          }
          else {
            auto& temp = temps.nextResource();
            auto offdiag_tile = comm::scheduleRecv(comm, rank_offdiag, tag_offdiag,
                                                   splitTile(temp(LocalTileIndex{0, 0}),
                                                             {{0, 0}, dist_a.tileSize(index_offdiag)}));
            dep = copy_offdiag(a_ws[id_block_local], k * nb,
                               ex::when_all(std::move(dep), std::move(offdiag_tile))) |
                  ex::split();
          }
        }

        deps[id_block_local].push_back(dep);
      }
      else {
        if (rank == rank_diag) {
          ex::start_detached(
              comm::scheduleSend(comm, rank_block, tag_diag, mat_a.read_sender(index_diag)));
        }
        if (k < nrtile - 1) {
          if (rank == rank_offdiag) {
            ex::start_detached(
                comm::scheduleSend(comm, rank_block, tag_offdiag, mat_a.read_sender(index_offdiag)));
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
  matrix::Distribution dist_panel({dist.localSize().cols(), b}, {nb, b});
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> v_panels(n_workspaces, dist_panel);

  auto init_sweep = [](std::shared_ptr<BandBlock<T, true>> a_block, SizeType sweep,
                       PromiseGuard<SweepWorkerDist<T>> worker) {
    worker.ref().startSweep(sweep, *a_block);
  };
  auto cont_sweep = [b](std::shared_ptr<BandBlock<T, true>> a_block, SizeType nr_steps,
                        PromiseGuard<SweepWorkerDist<T>> worker, matrix::Tile<T, Device::CPU>&& tile_v,
                        TileElementIndex index) {
    for (SizeType j = 0; j < nr_steps; ++j) {
      worker.ref().compactCopyToTile(tile_v, index + TileElementSize(j * b, 0));
      worker.ref().doStep(*a_block);
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

      if (auto n1 = a_block->nextSplit(start); n1 < n_d) {
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
                                  mat_trid.readwrite_sender(GlobalTileIndex{tile_index, 0}),
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
          ex::start_detached(scheduleSendCol(comm, prev_rank, compute_col_tag(block_id.col()), b,
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
          ex::start_detached(scheduleRecvWorker(sweep, init_step, comm, prev_rank,
                                                compute_worker_tag(sweep, block_id.col()),
                                                w_pipeline()));
        }

        // Index of the first column (currently) after this block (if exists).
        const SizeType next_j = sweep + (init_step + steps_per_block) * b;
        if (next_j < size) {
          // The dependency on the operation of the previous sweep is real as the Worker cannot be sent
          // before deps_block.back() gets ready, and the Worker is needed in the next rank to update the
          // column before is sent here.
          deps_block.push_back(scheduleRecvCol(comm, next_rank, compute_col_tag(block_id.col()), b,
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
                                          v_panel.readwrite_sender(local_index_tile_panel_v),
                                          TileElementIndex{0, sweep % b}, deps_block[dep_index]) |
              dlaf::internal::transform(policy_hp, cont_sweep) | ex::split();

          last_step = block_step;
        }

        // Shrink the dependency vector to only include the futures generated by this block in this sweep.
        deps_block.resize(ceilDiv(last_step, steps_per_task) + 1);

        if (init_step + block_steps < steps) {
          ex::start_detached(scheduleSendWorker(comm, next_rank,
                                                compute_worker_tag(sweep, block_id.col() + 1),
                                                w_pipeline()));
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
                                 const matrix::SubTileSpec spec_v) {
                  auto tile_v_panel = ex::keep_future(splitTile(v_panel.read(index_panel), spec_panel));
                  if (rank == rank_v) {
                    auto tile_v = splitTile(mat_v(index_v), spec_v);
                    ex::start_detached(ex::when_all(std::move(tile_v_panel), std::move(tile_v)) |
                                       copy(Policy<CopyBackend_v<Device::CPU, D>>{}));
                  }
                  else {
                    ex::start_detached(comm::scheduleSend(comm, rank_v, compute_v_tag(index_v.row()),
                                                          std::move(tile_v_panel)));
                  }
                };

            copy_or_send(helper.indexPanel(), helper.specPanelTop(), helper.rankVTop(),
                         helper.indexVTop(), helper.specVTop());
            if (helper.copyIsSplitted()) {
              copy_or_send(helper.indexPanel(), helper.specPanelBottom(), helper.rankVBottom(),
                           helper.indexVBottom(), helper.specVBottom());
            }
          }
          else {
            auto recv = [&comm, rank, &dist_v, &mat_v,
                         &compute_v_tag](const comm::IndexT_MPI rank_panel,
                                         const comm::IndexT_MPI rank_v, const GlobalTileIndex index_v,
                                         const matrix::SubTileSpec spec_v) {
              if (rank == rank_v) {
                auto tile_v = splitTile(mat_v(index_v), spec_v);
                auto local_index_v = dist_v.localTileIndex(index_v);

                ex::any_sender<> dep;
                if (local_index_v.col() == 0)
                  dep = ex::just();
                else
                  dep = ex::drop_value(mat_v.read_sender(local_index_v - LocalTileSize{0, 1}));

                ex::start_detached(comm::scheduleRecv(comm, rank_panel, compute_v_tag(index_v.row()),
                                                      ex::when_all(std::move(tile_v), std::move(dep))));
              }
            };

            recv(helper.rankPanel(), helper.rankVTop(), helper.indexVTop(), helper.specVTop());
            if (helper.copyIsSplitted()) {
              recv(helper.rankPanel(), helper.rankVBottom(), helper.indexVBottom(),
                   helper.specVBottom());
            }
          }
        }
      }
    }
  }

  // Rank 0 (owner of the first band matrix block) copies the last parts of the tridiag. matrix.
  if (rank == 0) {
    // copy the last elements of the diagonals
    if (!isComplex_v<T>) {
      // only needed for real types as they don't perform sweep size-2
      copy_tridiag(a_ws[0], size - 2, deps[0][0]);
    }
    copy_tridiag(a_ws[0], size - 1, std::move(deps[0][0]));
  }

  // only rank0 has mat_trid -> bcast to everyone.
  for (const auto& index : iterate_range2d(mat_trid.nrTiles())) {
    if (rank == 0)
      ex::start_detached(comm::scheduleSendBcast(comm_bcast(), mat_trid.read_sender(index)));
    else
      ex::start_detached(comm::scheduleRecvBcast(comm_bcast(), 0, mat_trid.readwrite_sender(index)));
  }

  return {std::move(mat_trid), std::move(mat_v)};
}
}
