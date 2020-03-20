#include <dlaf/matrix.h>

namespace dlaf {

DLAF_MATRIX_ETI_INST(float, Device::CPU)
DLAF_MATRIX_ETI_INST(double, Device::CPU)
DLAF_MATRIX_ETI_INST(std::complex<float>, Device::CPU)
DLAF_MATRIX_ETI_INST(std::complex<double>, Device::CPU)

// DLAF_MATRIX_ETI_INST(float, Device::GPU)
// DLAF_MATRIX_ETI_INST(double, Device::GPU)
// DLAF_MATRIX_ETI_INST(std::complex<float>, Device::GPU)
// DLAF_MATRIX_ETI_INST(std::complex<double>, Device::GPU)

}
