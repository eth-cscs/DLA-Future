#include "dlaf/eigensolver/gen_to_std/impl.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

DLAF_GENTOSTD_ETI(, Backend::GPU, Device::GPU, float)
DLAF_GENTOSTD_ETI(, Backend::GPU, Device::GPU, double)
DLAF_GENTOSTD_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_GENTOSTD_ETI(, Backend::GPU, Device::GPU, std::complex<double>)

}
}
}
