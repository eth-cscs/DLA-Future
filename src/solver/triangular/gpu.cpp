#include "dlaf/solver/triangular/impl.h"

namespace dlaf {
namespace solver {
namespace internal {

DLAF_SOLVER_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, float)
DLAF_SOLVER_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, double)
DLAF_SOLVER_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_SOLVER_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, std::complex<double>)
}
}
}
