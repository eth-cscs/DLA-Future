#pragma once

#include <array>

namespace dlaf {
namespace comm {
namespace test {

/// Compute valid 2D grid dimensions for a given number of ranks.
///
/// @return std::array<int, 2> an array with the two dimensions.
/// @post ret_dims[0] * ret_dims[0] == @p nranks.
std::array<int, 2> computeGridDims(int nranks);

}
}
}
