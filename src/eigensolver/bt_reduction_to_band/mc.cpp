#include "dlaf/eigensolver/bt_reduction_to_band.h"

namespace dlaf::eigensolver {

DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(, Backend::MC, Device::CPU, std::complex<double>)

}
