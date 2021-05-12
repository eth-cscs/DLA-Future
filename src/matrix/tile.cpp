#include <dlaf/matrix/tile.h>

namespace dlaf {
namespace matrix {

DLAF_TILE_ETI(, float, Device::CPU)
DLAF_TILE_ETI(, double, Device::CPU)
DLAF_TILE_ETI(, std::complex<float>, Device::CPU)
DLAF_TILE_ETI(, std::complex<double>, Device::CPU)

#if defined(DLAF_WITH_CUDA)
DLAF_TILE_ETI(, float, Device::GPU)
DLAF_TILE_ETI(, double, Device::GPU)
DLAF_TILE_ETI(, std::complex<float>, Device::GPU)
DLAF_TILE_ETI(, std::complex<double>, Device::GPU)
#endif
}
}
