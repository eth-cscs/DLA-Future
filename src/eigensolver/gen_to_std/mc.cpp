#include "dlaf/eigensolver/gen_to_std/impl.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

DLAF_GENTOSTD_ETI(, Backend::MC, Device::CPU, float)
DLAF_GENTOSTD_ETI(, Backend::MC, Device::CPU, double)
DLAF_GENTOSTD_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_GENTOSTD_ETI(, Backend::MC, Device::CPU, std::complex<double>)

}
}
}
