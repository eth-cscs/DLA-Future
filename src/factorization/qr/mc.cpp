#include "dlaf/factorization/qr.h"

namespace dlaf::factorization::internal {

DLAF_FACTORIZATION_QR_TFACTOR_ETI(, Backend::MC, Device::CPU, float)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(, Backend::MC, Device::CPU, double)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(, Backend::MC, Device::CPU, std::complex<double>)
}
