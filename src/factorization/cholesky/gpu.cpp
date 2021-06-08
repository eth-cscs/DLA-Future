#include "dlaf/factorization/cholesky/impl.h"

namespace dlaf {
namespace factorization {
namespace internal {

DLAF_FACTORIZATION_CHOLESKY_ETI(, Backend::GPU, Device::GPU, float)
DLAF_FACTORIZATION_CHOLESKY_ETI(, Backend::GPU, Device::GPU, double)
DLAF_FACTORIZATION_CHOLESKY_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_FACTORIZATION_CHOLESKY_ETI(, Backend::GPU, Device::GPU, std::complex<double>)
}
}
}
