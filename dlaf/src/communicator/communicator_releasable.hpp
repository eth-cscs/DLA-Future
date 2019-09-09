#pragma once

namespace dlaf {
namespace comm {

/// It exposes the release method of Communicator that it is kept protected to not allow the user
/// to release a Communicator.
struct CommunicatorReleasable : Communicator {
  void release() noexcept(false) { Communicator::release(); }
};

/// internal helper function to release communicator
auto release_communicator = [](Communicator comm) {
  static_cast<CommunicatorReleasable*>(&comm)->release();
};

}
}
