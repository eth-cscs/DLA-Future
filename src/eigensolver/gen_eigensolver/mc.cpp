#include "dlaf/eigensolver/gen_eigensolver/impl.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

DLAF_EIGENSOLVER_GEN_ETI(, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_GEN_ETI(, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_GEN_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_GEN_ETI(, Backend::MC, Device::CPU, std::complex<double>)

}
}
}
