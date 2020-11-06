#include "dlaf_test/util_mpi.h"

#include <mpi.h>

namespace dlaf {
namespace comm {
namespace test {

std::array<int, 2> computeGridDims(int nranks) {
  std::array<int, 2> dimensions{0, 0};
  MPI_Dims_create(nranks, 2, dimensions.data());
  return dimensions;
}

}
}
}
