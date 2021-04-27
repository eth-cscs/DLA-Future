//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

/// Panel (1D array of tiles)
///
/// 1D array of tiles, i.e. a Row or Column panel strictly related to a given dlaf::Matrix (from the
/// coords point of view)
template <Coord axis, class T, Device device>
struct Panel;

template <Coord axis, class T, Device device>
struct Panel<axis, const T, device> : protected Matrix<T, device> {
  // Note:
  // This specialization acts as base for the RW version of the panel,
  // moreover allows the casting between references (i.e. Panel<const T>& = Panel<T>)

  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;
  using BaseT = Matrix<T, device>;

  Panel(Panel&&) = default;

  /// On destruction, reset the panel
  ///
  /// Resetting the panel implies removing external dependencies
  virtual ~Panel() noexcept {
    reset();
  }

  /// Return an IterableRange2D with a range over all tiles of the panel (considering the offset)
  auto iterator() const noexcept {
    return range_;
  }

  /// Return the rank which this (local) panel belongs to
  auto rankIndex() const noexcept {
    return dist_matrix_.rankIndex();
  }

  /// Return the Distribution of the parent matrix
  auto parentDistribution() const noexcept {
    return dist_matrix_;
  }

  /// Set a specific index to point to the specified external tile
  ///
  /// It is possible to set to an external tile on an index if, since last reset() or from
  /// the creation of the panel, the specific index:
  /// - has not been accessed, neither on read or read/write
  /// - has not been already set to an external tile
  ///
  /// @pre @p index must be a valid index for the current panel size
  void setTile(const LocalTileIndex& index, hpx::shared_future<ConstTileT> new_tile_fut) {
    DLAF_ASSERT(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());
    DLAF_ASSERT(internal_.count(linearIndex(index)) == 0, "internal tile have been already used", index);
    DLAF_ASSERT(!isExternal(index), "already set to external", index);

    external_[linearIndex(index)] = std::move(new_tile_fut);
  }

  /// Access a Tile of the panel in read-only mode
  ///
  /// This method is very similar to the one available in dlaf::Matrix.
  ///
  /// @p index is in the coordinate system of the matrix which this panel is related to
  hpx::shared_future<ConstTileT> read(const LocalTileIndex& index) {
    DLAF_ASSERT_HEAVY(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());

    const SizeType internal_linear_idx = linearIndex(index);
    if (isExternal(index)) {
      return external_[internal_linear_idx];
    }
    else {
      internal_.insert(internal_linear_idx);
      return BaseT::read(fullIndex(index));
    }
  }

  /// Set the panel to a new offset (with respect to the "parent" matrix)
  ///
  /// @pre offset cannot be less than the offset has been specifed on construction
  void setOffset(LocalTileSize offset) noexcept {
    DLAF_ASSERT(offset.get(component(axis)) >= bias_, offset, bias_);

    offset_ = offset.get(component(axis)) - bias_;

    const LocalTileIndex panel_start(component(axis), offset_ + bias_);
    const LocalTileSize panel_size(component(axis),
                                   dist_matrix_.localNrTiles().get(component(axis)) - (offset_ + bias_),
                                   1);
    range_ = iterate_range2d(panel_start, panel_size);
  }

  /// Return the current set offset (1D)
  SizeType offset() const noexcept {
    return offset_;
  }

  /// Reset the internal usage status of the panel.
  ///
  /// In particular:
  /// - usage status of each tile is reset
  /// - external tiles references are dropped and internal ones are set back
  void reset() noexcept {
    for (auto& e : external_)
      e = {};
    internal_.clear();
  }

protected:
  using iter2d_t = decltype(iterate_range2d(LocalTileSize{0, 0}));

  static LocalElementSize computePanelSize(LocalElementSize size, TileElementSize blocksize,
                                           LocalTileSize start) {
    const auto mb = blocksize.rows();
    const auto nb = blocksize.cols();

    const auto mat_size = size.get(component(axis));
    const auto i_tile = start.get(component(axis));

    switch (axis) {
      case Coord::Col:
        return {mat_size - i_tile * mb, nb};
      case Coord::Row:
        return {mb, mat_size - i_tile * nb};
    }
  }

  /// Create the internal matrix, with tile layout, used for storing tiles
  ///
  /// It allocates just the memory needed for the part of matrix used, so
  /// starting from @p start
  static Matrix<T, device> setupMatrix(const Distribution& dist_matrix, const LocalTileSize start) {
    const auto panel_size = computePanelSize(dist_matrix.localSize(), dist_matrix.blockSize(), start);

    Distribution dist{panel_size, dist_matrix.blockSize()};
    auto layout = tileLayout(dist);
    return {std::move(dist), layout};
  }

  /// Create a Panel related to the Matrix passed as parameter.
  ///
  /// The Panel is strictly related to its parent dlaf::Matrix.
  /// In particular, it will create a Row or Column with the same size of its parent matrix (local),
  /// considering the specified offset from the top left origin.
  ///
  /// e.g. a 4x5 matrix with an offset 2x1 will have either:
  /// - a Panel<Col> 2x1
  /// - or a Panel<Row> 4x1
  Panel(matrix::Distribution dist_matrix, LocalTileSize offset)
      : BaseT(setupMatrix(dist_matrix, offset)), dist_matrix_(dist_matrix),
        bias_(offset.get(component(axis))), range_(iterate_range2d(LocalTileSize{0, 0})) {
    DLAF_ASSERT_HEAVY(BaseT::nrTiles().get(axis) == 1, BaseT::nrTiles());

    setOffset(offset);

    external_.resize(BaseT::nrTiles().get(component(axis)));

    DLAF_ASSERT_HEAVY(BaseT::distribution().localNrTiles().linear_size() == external_.size(),
                      BaseT::distribution().localNrTiles().linear_size(), external_.size());
  }

  /// Given a matrix index, compute the internal linear index
  SizeType linearIndex(const LocalTileIndex& index) const noexcept {
    const auto idx = index.get(component(axis));

    DLAF_ASSERT_MODERATE(idx >= offset_, idx, offset_);

    return idx - bias_;
  }

  /// Given a matrix index, compute the projected internal index
  ///
  /// It is similar to what linear_index does, so it takes into account the @p offset_,
  /// but it computes a 2D index instead of a linear one.
  /// The 2D index is the projection of the given index, i.e. in a Panel<Col> the Col for index
  /// will always be 0 (and relatively for a Panel<Row>)
  LocalTileIndex fullIndex(LocalTileIndex index) const {
    index = LocalTileIndex(component(axis), linearIndex(index));

    DLAF_ASSERT_HEAVY(index.isIn(BaseT::distribution().localNrTiles()), index,
                      BaseT::distribution().localNrTiles());

    return index;
  }

  /// Given a matrix index, check if the corresponding tile in the panel is external or not
  bool isExternal(const LocalTileIndex idx_matrix) const noexcept {
    return external_[linearIndex(idx_matrix)].valid();
  }

  ///> Parent matrix which this panel is related to
  Distribution dist_matrix_;

  ///> It represents from where it is necessary to allocate memory
  SizeType bias_;
  ///> It represents from where the panel gives access to tiles
  SizeType offset_;
  ///> Interanlly store the range of tiles accessible (changes according to @p offset_)
  iter2d_t range_;

  ///> Container for references to external tiles
  common::internal::vector<hpx::shared_future<ConstTileT>> external_;
  ///> Keep track of usage status of internal tiles (accessed or not)
  std::set<SizeType> internal_;
};

template <Coord axis, class T, Device device>
struct Panel : public Panel<axis, const T, device> {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  explicit Panel(matrix::Distribution distribution, LocalTileSize start = {0, 0})
      : Panel<axis, const T, device>(std::move(distribution), std::move(start)) {}

  /// Access tile at specified index in readwrite mode
  ///
  /// It is possible to access just internal tiles in RW mode.
  ///
  /// @pre index must point to a tile which is internally managed by the panel
  hpx::future<TileT> operator()(const LocalTileIndex& index) {
    DLAF_ASSERT(!isExternal(index), "read-only access on external tiles", index);

    BaseT::internal_.insert(BaseT::linearIndex(index));
    return BaseT::operator()(BaseT::fullIndex(index));
  }

protected:
  using BaseT = Panel<axis, const T, device>;
  using BaseT::isExternal;
};

namespace internal {

// helper function that identifies the owner of a transposed coordinate,
// it returns both the component of the rank in the transposed dimension and
// its global cross coordinate (i.e. row == col in the global frame of reference)
template <Coord dst_coord>
std::pair<SizeType, comm::IndexT_MPI> transposedOwner(const Distribution& dist,
                                                      const LocalTileIndex idx) {
  const auto idx_cross = dist.template globalTileFromLocalTile<dst_coord>(idx.get(dst_coord));
  const auto rank_owner = dist.template rankGlobalTile<orthogonal(dst_coord)>(idx_cross);
  return std::make_pair(idx_cross, rank_owner);
}
}

/// Broadcast
///
/// Given a source panel on a rank, it gets broadcasted to make it available to all other ranks.
///
/// It does not give access to all the tiles, but just the ones of interest for each rank.
///
/// @param rank_root    on which rank the @p panel contains data to be broadcasted
/// @param panel        on @p rank_root it is the source panel
///                     on other ranks it is the destination panel
/// @param serial_comm  where to pipeline the tasks for communications.
/// @pre Communicator in @p serial_comm must be orthogonal to panel axis
template <class T, Device device, Coord axis, class = std::enable_if_t<!std::is_const<T>::value>>
void broadcast(const comm::Executor& ex, comm::IndexT_MPI rank_root, Panel<axis, T, device>& panel,
               common::Pipeline<comm::Communicator>& serial_comm, const comm::Size2D grid_size) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  constexpr auto comm_dir = orthogonal(axis);

  // do not schedule communication tasks if there is no reason to do so...
  if (grid_size.get(component(comm_dir)) <= 1)
    return;

  const auto rank = panel.rankIndex().get(component(comm_dir));

  for (const auto& index : panel.iterator()) {
    if (rank == rank_root)
      dataflow(ex, unwrapping(comm::sendBcast<T>), panel.read(index), serial_comm());
    else
      dataflow(ex, unwrapping(comm::recvBcast<T>), panel(index), rank_root, serial_comm());
  }
}

/// Broadcast
///
/// Given a source panel on a rank, this communication pattern makes every rank access tiles of both:
/// a. the source panel
/// b. it's tranposed variant (just tile coordinates, data is not transposed) w.r.t. the main diagonal of
/// the parent matrix
//
/// In particular, it does not give access to all the tiles, but just the ones of interest for
/// each rank, i.e. the rows and columns of a distributed matrix that the ranks stores locally.
///
/// This is achieved by either:
/// - linking as external tile, if the tile is already available locally for the rank
/// - receiving the tile from the owning rank (via a broadcast)
///
/// @param rank_root specifies on which rank the @p panel is the source of the data
/// @param panel
///   on rank_root it is the source panel (a)
///   on others it represents the destination for the broadcast (b)
/// @param panelT it represents the destination panel for the "transposed" variant of the panel
/// @param row_task_chain where to pipeline the tasks for row-wise communications
/// @param col_task_chain where to pipeline the tasks for col-wise communications
/// @param grid_size shape of the grid of row and col communicators from @p row_task_chain and @p col_task_chain
///
/// @pre both panels are child of a matrix (even not the same) with the same Distribution
/// @pre both panels parent matrices should be square matrices with square blocksizes
/// @pre both panels offsets should lay on the main diagonal of the parent matrix
template <class T, Device device, Coord axis, class = std::enable_if_t<!std::is_const<T>::value>>
void broadcast(const comm::Executor& ex, comm::IndexT_MPI rank_root, Panel<axis, T, device>& panel,
               Panel<orthogonal(axis), T, device>& panelT,
               common::Pipeline<comm::Communicator>& row_task_chain,
               common::Pipeline<comm::Communicator>& col_task_chain, comm::Size2D grid_size) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  constexpr Coord axisT = orthogonal(axis);

  auto get_taskchain = [&](Coord comm_dir) -> auto& {
    return comm_dir == Coord::Row ? row_task_chain : col_task_chain;
  };

  // Note:
  // Given a source panel, this communication pattern makes every rank access tiles of both the
  // source panel and it's tranposed variant (just tile coordinates, data is not transposed).
  // In particular, it does not give access to all the tiles, but just the ones of interest for
  // each rank, i.e. the rows and columns of a distributed matrix that the ranks stores locally.
  //
  // This happens in two steps (for the sake of example, let's consider a column -> row broadcast,
  // the opposite is dual):
  //
  // 1. broadcast the source panel to panel with the same shape on other ranks
  // 2. populate the transposed destination panel
  //
  // Once the source panel is share by all ranks, the transposed panel can be easily populated,
  // because each destination tile can be populated with data from the rank owning the diagonal one,
  // the point of contact between the row and the column (row == col).
  //
  // If it is already available locally, the tile is not copied and it just gets "linked", in order
  // to easily access it via panel coordinates with minimal (to null) overhead.
  // For this reason, the destination panel will depend on the source panel (as the source panel
  // may already depend on the matrix).

  DLAF_ASSERT(panel.parentDistribution() == panelT.parentDistribution(),
              "they must refer to the same matrix");

  const auto& dist = panel.parentDistribution();

  // Note:
  // This algorithm allow to broadcast panel to panelT using as mirror the parent matrix main diagonal.
  // This means that it is possible to broadcast panels with different axes just if their global offset
  // lie on the diaognal.
  // In order to verify this, a check is performed by verifying on each rank what are the possible
  // indices for the global offset, starting from the local one.
  //
  // Given the distribution and the local offset, a set of possible indices is built, considering
  // the follow inequalty:
  // globalFromLocal(offset_local) - grid_size < offset_global <= globalFromLocal(offset_local)
  // and producing a list of `grid_size` possible values in each panel direction
  //
  // At this point, the check verifies that there is at least a match among the two sets.
  // If all ranks have at least a matching global offset among the two different directions,
  // it means that the global offset for the panels is on the main diagonal.
  //
  // credits: @rasolca
  DLAF_ASSERT(square_size(dist), dist.size());
  DLAF_ASSERT(square_blocksize(dist), dist.blockSize());
  DLAF_ASSERT_MODERATE(
      [&]() {
        const auto offset = dist.template globalTileFromLocalTile<component(axis)>(panel.offset());
        const auto offsetT = dist.template globalTileFromLocalTile<component(axisT)>(panelT.offset());

        const auto grid_size = dist.commGridSize().get(component(axis));
        const auto gridT_size = dist.commGridSize().get(component(axisT));

        auto generate_indices = [](SizeType offset, SizeType grid_size) {
          std::vector<SizeType> indices(to_sizet(grid_size));
          std::iota(indices.begin(), indices.end(), offset - grid_size + 1);
          return indices;
        };

        std::vector<SizeType> indices = generate_indices(offset, grid_size);
        std::vector<SizeType> indicesT = generate_indices(offsetT, gridT_size);

        std::vector<SizeType> common_indices(std::min(indices.size(), indicesT.size()));
        const auto chances =
            std::distance(common_indices.begin(),
                          std::set_intersection(indices.begin(), indices.end(), indicesT.begin(),
                                                indicesT.end(), common_indices.begin()));

        return chances > 0;
      }(),
      panel.offset(), panelT.offset(), "broadcast can mirror just on the parent matrix main diagonal");

  // STEP 1
  constexpr auto comm_dir_step1 = orthogonal(axis);
  auto& chain_step1 = get_taskchain(comm_dir_step1);

  broadcast(ex, rank_root, panel, chain_step1, grid_size);

  // STEP 2
  constexpr Coord coord = component(axis);
  constexpr Coord coordT = component(axisT);

  constexpr auto comm_dir_step2 = orthogonal(axisT);
  auto& chain_step2 = get_taskchain(comm_dir_step2);

  for (const auto& indexT : panelT.iterator()) {
    SizeType index_diag;
    comm::IndexT_MPI owner_diag;

    std::tie(index_diag, owner_diag) = internal::transposedOwner<coordT>(dist, indexT);

    if (dist.rankIndex().get(coord) == owner_diag) {
      const auto index_diag_local = dist.template localTileFromGlobalTile<coord>(index_diag);
      panelT.setTile(indexT, panel.read({coord, index_diag_local}));

      if (grid_size.get(component(comm_dir_step2)) > 1)
        dataflow(ex, unwrapping(comm::sendBcast<T>), panelT.read(indexT), chain_step2());
    }
    else {
      if (grid_size.get(component(comm_dir_step2)) > 1)
        dataflow(ex, unwrapping(comm::recvBcast<T>), panelT(indexT), owner_diag, chain_step2());
    }
  }
}
}
}
