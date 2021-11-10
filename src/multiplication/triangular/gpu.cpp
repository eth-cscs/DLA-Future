#include "dlaf/multiplication/triangular/impl.h"

namespace dlaf {
namespace multiplication {
namespace internal {

DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, float)
DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, double)
DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, std::complex<double>)
}
}
}
