#include "dlaf/eti/dataflow.h"

namespace dlaf {

DLAF_DATAFLOW_ETI(, float, Device::CPU)
DLAF_DATAFLOW_ETI(, double, Device::CPU)
DLAF_DATAFLOW_ETI(, std::complex<float>, Device::CPU)
DLAF_DATAFLOW_ETI(, std::complex<double>, Device::CPU)

}
