#include "dlaf/eigensolver/reduction_to_band.h"

namespace dlaf::eigensolver {

DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(, Backend::GPU, Device::GPU, float)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(, Backend::GPU, Device::GPU, double)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(, Backend::GPU, Device::GPU, std::complex<double>)

}
