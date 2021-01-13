#include "dlaf/solver/triangular/impl.h"

namespace dlaf {
namespace solver {
namespace internal {

DLAF_SOLVER_TRIANGULAR_ETI(, Backend::MC, Device::CPU, float)
DLAF_SOLVER_TRIANGULAR_ETI(, Backend::MC, Device::CPU, double)
DLAF_SOLVER_TRIANGULAR_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_SOLVER_TRIANGULAR_ETI(, Backend::MC, Device::CPU, std::complex<double>)
}
}
}
