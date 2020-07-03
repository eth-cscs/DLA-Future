#include <dlaf/communication/executor.h>

namespace dlaf {
namespace comm {

// initialize static variable
std::atomic<int> executor::num_pending_comms(0);

}
}
