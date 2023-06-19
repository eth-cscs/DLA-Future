#include "test_grid_c_api_wrapper.h"

#include <dlaf_c/grid.h>

char C_grid_ordering(MPI_Comm comm, int nprow, int npcol, int myprow, int mypcol)
{
    return grid_ordering(comm, nprow, npcol, myprow, mypcol);
}
