#include <dlaf/memory/memory_view.h>

namespace dlaf {
namespace memory {

DLAF_MEMVIEW_ETI_INST(float, Device::CPU)
DLAF_MEMVIEW_ETI_INST(double, Device::CPU)
DLAF_MEMVIEW_ETI_INST(std::complex<float>, Device::CPU)
DLAF_MEMVIEW_ETI_INST(std::complex<double>, Device::CPU)

// DLAF_MEMVIEW_ETI_INST(float, Device::GPU)
// DLAF_MEMVIEW_ETI_INST(double, Device::GPU)
// DLAF_MEMVIEW_ETI_INST(std::complex<float>, Device::GPU)
// DLAF_MEMVIEW_ETI_INST(std::complex<double>, Device::GPU)

}
}
