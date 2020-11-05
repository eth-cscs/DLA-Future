#include <dlaf/matrix.h>

namespace dlaf {
namespace matrix {

DLAF_MATRIX_ETI(, float, Device::CPU)
DLAF_MATRIX_ETI(, double, Device::CPU)
DLAF_MATRIX_ETI(, std::complex<float>, Device::CPU)
DLAF_MATRIX_ETI(, std::complex<double>, Device::CPU)

// DLAF_MATRIX_ETI(, float, Device::GPU)
// DLAF_MATRIX_ETI(, double, Device::GPU)
// DLAF_MATRIX_ETI(, std::complex<float>, Device::GPU)
// DLAF_MATRIX_ETI(, std::complex<double>, Device::GPU)

}
}
