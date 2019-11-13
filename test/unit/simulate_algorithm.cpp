#include <gtest/gtest.h>

#include "dlaf/common/pipeline.h"

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/matrix.h"

#include "dlaf_test/util_tile.h"

using namespace dlaf;
using dlaf::common::Pipeline;

TEST(Pipeline, Basic) {
  static_assert(NUM_MPI_RANKS == 2, "This test requires exactly 2 MPI ranks");
  using TypeParam = float;

  matrix::LayoutInfo layout = matrix::colMajorLayout({10, 10}, {2, 2}, 10);
  memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
  TypeParam* p = mem();
  auto matrix = createMatrixFromColMajor<Device::CPU>({10, 10}, {2, 2}, 10, p, mem.size());

  dlaf_test::tile_test::set(matrix({0, 0}).get(), 0);

  dlaf::comm::Communicator world(MPI_COMM_WORLD);
  int rank = world.rank();

  Pipeline<dlaf::comm::Communicator> serial_comm(std::move(world));

  if (rank == 0) {
    matrix({0, 0}).then(hpx::util::unwrapping([](auto&& tile) {
      CHECK_TILE_EQ(0, tile);
      dlaf_test::tile_test::set(tile, 13);
      CHECK_TILE_EQ(13, tile);
      return std::move(tile);
    }));

    hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                    dlaf::comm::sync::broadcast::send(comm_wrapper(), tile);
                    CHECK_TILE_EQ(13, tile);
                  }),
                  matrix.read({0, 0}), serial_comm());
  }
  else {
    auto last = serial_comm().then(hpx::util::unwrapping([](auto&& comm_wrapper) mutable {
      matrix::LayoutInfo layout = matrix::colMajorLayout({2, 2}, {2, 2}, 2);

      dlaf::Tile<TypeParam, dlaf::Device::CPU> workspace({2, 2}, {layout.minMemSize()}, 2);
      dlaf::comm::sync::broadcast::receive_from(0, comm_wrapper(), workspace);
      return std::move(workspace);
    }));

    last.then(hpx::util::unwrapping([](auto&& tile) { CHECK_TILE_EQ(13, tile); }));
  }
}
