#include <dlaf_c/utils.h>
#include <mpi.h>

DLAF_EXTERN_C char C_grid_ordering(MPI_Comm comm, int nprow, int npcol, int myprow, int mypcol);
