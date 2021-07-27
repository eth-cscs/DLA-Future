//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/reduction_to_band.h"

#include <cmath>

#include <gtest/gtest.h>
#include <lapack/util.hh>

#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class ReductionToBandTestMC : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(ReductionToBandTestMC, MatrixElementTypes);

struct config_t {
  LocalElementSize size;
  TileElementSize block_size;
};

std::vector<config_t> configs{
    {{3, 3}, {3, 3}},    // single tile (nothing to do)
    {{12, 12}, {3, 3}},  // tile always full size (less room for distribution over ranks)
    {{13, 13}, {3, 3}},  // tile incomplete
    {{24, 24}, {3, 3}},  // tile always full size (more room for distirbution)
};

template <class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().blockSize()};
}

template <class T>
void mirrorLowerOnDiag(const Tile<T, Device::CPU>& tile) {
  DLAF_ASSERT(square_size(tile), tile.size());

  for (SizeType j = 0; j < tile.size().cols(); j++)
    for (SizeType i = j; i < tile.size().rows(); ++i)
      tile({j, i}) = dlaf::conj(tile({i, j}));
}

template <class T>
void copyConjTrans(const Tile<const T, Device::CPU>& from, const Tile<T, Device::CPU>& to) {
  DLAF_ASSERT(from.size() == transposed(to.size()), from.size(), to.size());

  for (SizeType j = 0; j < from.size().cols(); j++)
    for (SizeType i = 0; i < from.size().rows(); ++i)
      to({j, i}) = dlaf::conj(from({i, j}));
}

template <class T>
void setupHermitianBand(MatrixLocal<T>& matrix, const SizeType& band_size) {
  DLAF_ASSERT(band_size == matrix.blockSize().rows(), band_size, matrix.blockSize().rows());
  DLAF_ASSERT(band_size % matrix.blockSize().rows() == 0, band_size, matrix.blockSize().rows());

  DLAF_ASSERT(square_blocksize(matrix), matrix.blockSize());
  DLAF_ASSERT(square_size(matrix), matrix.blockSize());

  const auto k_diag = band_size / matrix.blockSize().rows();

  // setup band "edge" and its tranposed
  for (SizeType j = 0; j < matrix.nrTiles().cols(); ++j) {
    const GlobalTileIndex ij{j + k_diag, j};

    if (!ij.isIn(matrix.nrTiles()))
      continue;

    const auto& tile_lo = matrix.tile(ij);

    // setup the strictly lower to zero (if there is any)
    if (std::min(tile_lo.size().rows(), tile_lo.size().cols()) - 1 > 0) {
      lapack::laset(lapack::MatrixType::Lower, tile_lo.size().rows() - 1, tile_lo.size().cols() - 1, 0,
                    0, tile_lo.ptr({1, 0}), tile_lo.ld());
    }

    copyConjTrans(matrix.tile_read(ij), matrix.tile(common::transposed(ij)));
  }

  // setup zeros in both lower and upper out-of-band
  for (SizeType j = 0; j < matrix.nrTiles().cols(); ++j) {
    for (SizeType i = j + k_diag + 1; i < matrix.nrTiles().rows(); ++i) {
      const GlobalTileIndex ij{i, j};
      if (!ij.isIn(matrix.nrTiles()))
        continue;

      tile::set0(matrix.tile(ij));
      tile::set0(matrix.tile(common::transposed(ij)));
    }
  }

  // mirror band (diagonal tiles are correctly set just in the lower part)
  for (SizeType k = 0; k < matrix.nrTiles().rows(); ++k) {
    const GlobalTileIndex kk{k, k};
    const auto& tile = matrix.tile(kk);

    mirrorLowerOnDiag(tile);
  }
}

template <class T>
void splitReflectorsAndBand(MatrixLocal<T>& mat_v, MatrixLocal<T>& mat_b, const SizeType band_size) {
  DLAF_ASSERT_HEAVY(square_size(mat_v), mat_v.size());
  DLAF_ASSERT_HEAVY(square_size(mat_b), mat_b.size());

  DLAF_ASSERT(equal_size(mat_v, mat_b), mat_v.size(), mat_b.size());

  for (SizeType diag = 0; diag <= 1; ++diag) {
    for (SizeType i = diag; i < mat_v.nrTiles().rows(); ++i) {
      const GlobalTileIndex idx(i, i - diag);
      copy(mat_v.tile(idx), mat_b.tile(idx));
    }
  }

  setupHermitianBand(mat_b, band_size);
}

template <class T>
auto allGatherTaus(const SizeType k, const SizeType chunk_size, const SizeType band_size,
                   std::vector<hpx::shared_future<common::internal::vector<T>>> fut_local_taus,
                   comm::CommunicatorGrid comm_grid) {
  std::vector<T> taus;
  taus.reserve(to_sizet(k));

  hpx::wait_all(fut_local_taus);
  auto local_taus = hpx::unwrap(fut_local_taus);

  DLAF_ASSERT(band_size == chunk_size, band_size, chunk_size);

  const auto n_chunks = std::ceil(float(k) / chunk_size);

  for (auto index_chunk = 0; index_chunk < n_chunks; ++index_chunk) {
    const auto owner = index_chunk % comm_grid.size().cols();
    const bool is_owner = owner == comm_grid.rank().col();

    const auto this_chunk_size = std::min(k - index_chunk * chunk_size, chunk_size);

    std::vector<T> chunk_data;
    if (is_owner) {
      const auto index_chunk_local = to_sizet(index_chunk / comm_grid.size().cols());
      chunk_data = local_taus[index_chunk_local];
      sync::broadcast::send(comm_grid.rowCommunicator(),
                            common::make_data(chunk_data.data(),
                                              static_cast<SizeType>(chunk_data.size())));
    }
    else {
      chunk_data.resize(to_sizet(this_chunk_size));
      sync::broadcast::receive_from(owner, comm_grid.rowCommunicator(),
                                    common::make_data(chunk_data.data(),
                                                      static_cast<SizeType>(chunk_data.size())));
    }

    // copy each chunk contiguously
    std::copy(chunk_data.begin(), chunk_data.end(), std::back_inserter(taus));
  }

  return taus;
}

TYPED_TEST(ReductionToBandTestMC, CorrectnessDistributed) {
  constexpr Device device = Device::CPU;

  for (auto&& comm_grid : this->commGrids()) {
    for (const auto& config : configs) {
      const LocalElementSize& size = config.size;
      const TileElementSize& block_size = config.block_size;

      const SizeType band_size = block_size.rows();
      const SizeType band_size_tiles = band_size / block_size.rows();
      const SizeType k_reflectors = std::max(SizeType(0), size.rows() - band_size - 1);

      Distribution distribution({size.rows(), size.cols()}, block_size, comm_grid.size(),
                                comm_grid.rank(), {0, 0});

      // setup the reference input matrix
      Matrix<const TypeParam, device> reference = [&]() {
        Matrix<TypeParam, device> reference(distribution);
        matrix::util::set_random_hermitian(reference);
        return reference;
      }();

      Matrix<TypeParam, device> matrix_a(std::move(distribution));
      copy(reference, matrix_a);

      // Apply reduction-to-band
      DLAF_ASSERT(band_size == matrix_a.blockSize().rows(), band_size, matrix_a.blockSize().rows());

      auto local_taus = eigensolver::reductionToBand<Backend::MC>(comm_grid, matrix_a);

      // First basic check: the reduction should not affect the strictly upper part of the input
      auto check_rest_unchanged = [&reference, &matrix_a](const GlobalElementIndex& index) {
        const auto& dist = reference.distribution();
        const auto ij_tile = dist.globalTileIndex(index);
        const auto ij_element_wrt_tile = dist.tileElementIndex(index);

        const bool is_in_upper = index.row() < index.col();

        if (!is_in_upper)
          return matrix_a.read(ij_tile).get()(ij_element_wrt_tile);
        else
          return reference.read(ij_tile).get()(ij_element_wrt_tile);
      };
      CHECK_MATRIX_NEAR(check_rest_unchanged, matrix_a, 0, TypeUtilities<TypeParam>::error);

      // The distributed result of reduction to band has the following characteristics:
      // - just lower part is relevant (diagonal is included)
      // - it stores the lower part of the band (B)
      // - and the matrix V with all the reflectors (the 1 component of each reflector is omitted)
      //
      // Each rank collects locally the full matrix by storing each part separately
      // - the B matrix, which is then "completed" by zeroing elements out of the band, and by mirroring the
      //    existing part of the band
      // - the V matrix, whose relevant part is the submatrix underneath the band

      auto mat_v = allGather(lapack::MatrixType::Lower, matrix_a, comm_grid);
      auto mat_b = makeLocal(matrix_a);
      splitReflectorsAndBand(mat_v, mat_b, band_size);

      // TODO FIXME mat_v can be smaller, but then allGather must copy a submatrix
      const GlobalTileIndex v_offset{band_size_tiles, 0};

      // Finally, collect locally tau values, which together with V allow to apply the Q transformation
      auto taus = allGatherTaus(k_reflectors, block_size.rows(), band_size, local_taus, comm_grid);
      DLAF_ASSERT(to_SizeType(taus.size()) == k_reflectors, taus.size(), k_reflectors);

      // Now that all input are collected locally, it's time to apply the transformation,
      // ...but just if there is any
      if (v_offset.isIn(mat_v.nrTiles())) {
        // Reduction to band returns a sequence of transformations applied from left and right to A
        // allowing to reduce the matrix A to a band matrix B
        //
        // Hn* ... H2* H1* A H1 H2 ... Hn
        // Q* A Q = B
        //
        // Applying the inverse of the same transformations, we can go from B to A
        // Q B Q* = A
        // Q = H1 H2 ... Hn
        // H1 H2 ... Hn B Hn* ... H2* H1*

        // apply from left...
        const GlobalTileIndex left_offset{band_size_tiles, 0};
        const GlobalElementSize left_size{mat_b.size().rows() - band_size, mat_b.size().cols()};
        lapack::unmqr(lapack::Side::Left, lapack::Op::NoTrans, left_size.rows(), left_size.cols(),
                      k_reflectors, mat_v.tile_read(v_offset).ptr(), mat_v.ld(), taus.data(),
                      mat_b.tile(left_offset).ptr(), mat_b.ld());

        // ... and from right
        const GlobalTileIndex right_offset = common::transposed(left_offset);
        const GlobalElementSize right_size = common::transposed(left_size);

        lapack::unmqr(lapack::Side::Right, lapack::Op::ConjTrans, right_size.rows(), right_size.cols(),
                      k_reflectors, mat_v.tile_read(v_offset).ptr(), mat_v.ld(), taus.data(),
                      mat_b.tile(right_offset).ptr(), mat_b.ld());
      }

      // Eventually, check the result obtained by applying the inverse transformation equals the original matrix
      auto result = [& dist = reference.distribution(),
                     &mat_local = mat_b](const GlobalElementIndex& element) {
        const auto tile_index = dist.globalTileIndex(element);
        const auto tile_element = dist.tileElementIndex(element);
        return mat_local.tile_read(tile_index)(tile_element);
      };
      CHECK_MATRIX_NEAR(result, reference, 0,
                        mat_b.size().linear_size() * TypeUtilities<TypeParam>::error);
    }
  }
}
