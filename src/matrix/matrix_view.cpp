#include <dlaf/matrix/matrix_view.h>

namespace dlaf {
namespace matrix {

DLAF_MATRIXVIEW_ETI(, float, Device::CPU)
DLAF_MATRIXVIEW_ETI(, double, Device::CPU)
DLAF_MATRIXVIEW_ETI(, std::complex<float>, Device::CPU)
DLAF_MATRIXVIEW_ETI(, std::complex<double>, Device::CPU)

// DLAF_MATRIXVIEW_ETI(, float, Device::GPU)
// DLAF_MATRIXVIEW_ETI(, double, Device::GPU)
// DLAF_MATRIXVIEW_ETI(, std::complex<float>, Device::GPU)
// DLAF_MATRIXVIEW_ETI(, std::complex<double>, Device::GPU)

}
}
