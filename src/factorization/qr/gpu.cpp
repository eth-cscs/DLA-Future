#include "dlaf/factorization/qr.h"

namespace dlaf::factorization::internal {

DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(, Backend::GPU, Device::GPU, float)
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(, Backend::GPU, Device::GPU, double)
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(, Backend::GPU, Device::GPU, std::complex<double>)
}
