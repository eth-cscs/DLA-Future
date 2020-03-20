#include <dlaf/tile.h>

namespace dlaf {

DLAF_TILE_ETI_INST(float, Device::CPU)
DLAF_TILE_ETI_INST(double, Device::CPU)
DLAF_TILE_ETI_INST(std::complex<float>, Device::CPU)
DLAF_TILE_ETI_INST(std::complex<double>, Device::CPU)

}
