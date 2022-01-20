#include <mkl.h>
#include <dlaf/common/assert.h>
#include <dlaf/lapack/laed4.h>

namespace dlaf {
namespace internal {

void laed4_wrapper(int n, int i, float const* d, float const* z, float* delta, float rho,
                   float* lambda) {
  ++i;  // Fortran indexing starts from 1
  int info = 0;
  slaed4(&n, &i, d, z, delta, &rho, lambda, &info);
  DLAF_ASSERT(info >= 0, info);
}

void laed4_wrapper(int n, int i, double const* d, double const* z, double* delta, double rho,
                   double* lambda) {
  ++i;  // Fortran indexing starts from 1
  int info = 0;
  dlaed4(&n, &i, d, z, delta, &rho, lambda, &info);
  DLAF_ASSERT(info >= 0, info);
}

}
}
