#include "dlaf/factorization/cholesky/gpu.h"

namespace dlaf {
namespace factorization {
namespace internal {

// TODO: Not available due to herk.
// DLAF_CHOLESKY_GPU_ETI(, float)
// DLAF_CHOLESKY_GPU_ETI(, double)
DLAF_CHOLESKY_GPU_ETI(, std::complex<float>)
DLAF_CHOLESKY_GPU_ETI(, std::complex<double>)

}
}
}
