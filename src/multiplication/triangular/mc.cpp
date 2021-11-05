#include "dlaf/multiplication/triangular/impl.h"

namespace dlaf {
namespace multiplication {
namespace internal {

DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::MC, Device::CPU, float)
DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::MC, Device::CPU, double)
DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::MC, Device::CPU, std::complex<double>)
}
}
}
