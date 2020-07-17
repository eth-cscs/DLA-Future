#pragma once

#include <array>

namespace dlaf_test {
namespace comm {

/// Compute valid 2D grid dimensions for a given number of ranks.
///
/// @return std::array<int, 2> an array with the two dimensions.
/// @post ret_dims[0] * ret_dims[0] == @p nranks.
std::array<int, 2> computeGridDims(int nranks);

}
}
