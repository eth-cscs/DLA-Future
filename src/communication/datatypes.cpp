#include <dlaf/communication/datatypes.h>

namespace dlaf {
namespace comm {

// clang-format off
template<> MPI_Datatype mpi_datatype<char>                 ::type = MPI_CHAR;
template<> MPI_Datatype mpi_datatype<short>                ::type = MPI_SHORT;
template<> MPI_Datatype mpi_datatype<int>                  ::type = MPI_INT;
template<> MPI_Datatype mpi_datatype<long>                 ::type = MPI_LONG;
template<> MPI_Datatype mpi_datatype<long long>            ::type = MPI_LONG_LONG;
template<> MPI_Datatype mpi_datatype<unsigned char>        ::type = MPI_UNSIGNED_CHAR;
template<> MPI_Datatype mpi_datatype<unsigned short>       ::type = MPI_UNSIGNED_SHORT;
template<> MPI_Datatype mpi_datatype<unsigned int>         ::type = MPI_UNSIGNED;
template<> MPI_Datatype mpi_datatype<unsigned long>        ::type = MPI_UNSIGNED_LONG;
template<> MPI_Datatype mpi_datatype<unsigned long long>   ::type = MPI_UNSIGNED_LONG_LONG;
template<> MPI_Datatype mpi_datatype<float>                ::type = MPI_FLOAT;
template<> MPI_Datatype mpi_datatype<double>               ::type = MPI_DOUBLE;
template<> MPI_Datatype mpi_datatype<bool>                 ::type = MPI_CXX_BOOL;
template<> MPI_Datatype mpi_datatype<std::complex<float>>  ::type = MPI_CXX_FLOAT_COMPLEX;
template<> MPI_Datatype mpi_datatype<std::complex<double>> ::type = MPI_CXX_DOUBLE_COMPLEX;
// clang-format on

}
}
