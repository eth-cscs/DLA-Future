#pragma once

#include <hpx/future.hpp>

namespace dlaf {
namespace cuda {
namespace internal {

// A CUDA mutex to protect CUBLAS and CUDA calls.
hpx::lcos::local::mutex& get_cuda_mtx();

}
}
}
