#include <dlaf/memory/memory_view.h>

namespace dlaf {
namespace memory {

DLAF_MEMVIEW_ETI(, float, Device::CPU)
DLAF_MEMVIEW_ETI(, double, Device::CPU)
DLAF_MEMVIEW_ETI(, std::complex<float>, Device::CPU)
DLAF_MEMVIEW_ETI(, std::complex<double>, Device::CPU)

// DLAF_MEMVIEW_ETI(, float, Device::GPU)
// DLAF_MEMVIEW_ETI(, double, Device::GPU)
// DLAF_MEMVIEW_ETI(, std::complex<float>, Device::GPU)
// DLAF_MEMVIEW_ETI(, std::complex<double>, Device::GPU)

}
}
