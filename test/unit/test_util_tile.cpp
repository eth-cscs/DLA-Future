#include "dlaf/util_tile.h"

#include <gtest/gtest.h>

#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf_test/util_types.h"

template <typename Type>
class TileUtilsTest : public ::testing::Test {};

TYPED_TEST_CASE(TileUtilsTest, dlaf_test::MatrixElementTypes);

TYPED_TEST(TileUtilsTest, CreateBuffer) {
  using namespace dlaf;

  SizeType m = 37;
  SizeType n = 87;
  SizeType ld = 133;

  memory::MemoryView<TypeParam, Device::CPU> memory_view(ld * n);
  auto mem_view = memory_view;

  TileElementSize size(m, n);
  Tile<TypeParam, Device::CPU> tile(size, std::move(mem_view), ld);

  auto tile_buffer = dlaf::common::make_buffer(tile);

  EXPECT_EQ(tile.ptr({0, 0}), get_pointer(tile_buffer));
  EXPECT_EQ(tile.size().cols(), get_num_blocks(tile_buffer));
  EXPECT_EQ(tile.size().rows(), get_blocksize(tile_buffer));
  EXPECT_EQ(tile.ld(), get_stride(tile_buffer));
}
