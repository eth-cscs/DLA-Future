#include "test_cholesky_c_api_wrapper.h"

#include <dlaf_c/factorization/cholesky.h>

void C_dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int* info){
  dlaf_pdpotrf(uplo, n, a, ia, ja, desca, info);
}
