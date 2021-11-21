#include "dlaf/eigensolver/tridiag_solver/mc.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(, Backend::MC, Device::CPU, float)
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(, Backend::MC, Device::CPU, double)
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(, Backend::MC, Device::CPU, std::complex<double>)

}
}
}
