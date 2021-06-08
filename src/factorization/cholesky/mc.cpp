#include "dlaf/factorization/cholesky/impl.h"

namespace dlaf {
namespace factorization {
namespace internal {

DLAF_FACTORIZATION_CHOLESKY_ETI(, Backend::MC, Device::CPU, float)
DLAF_FACTORIZATION_CHOLESKY_ETI(, Backend::MC, Device::CPU, double)
DLAF_FACTORIZATION_CHOLESKY_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_FACTORIZATION_CHOLESKY_ETI(, Backend::MC, Device::CPU, std::complex<double>)
}
}
}
