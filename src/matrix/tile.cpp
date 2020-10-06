#include <dlaf/matrix/tile.h>

namespace dlaf {
namespace matrix {

DLAF_TILE_ETI(, float, Device::CPU)
DLAF_TILE_ETI(, double, Device::CPU)
DLAF_TILE_ETI(, std::complex<float>, Device::CPU)
DLAF_TILE_ETI(, std::complex<double>, Device::CPU)

}
}
