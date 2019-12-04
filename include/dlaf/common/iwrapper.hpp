#pragma once

#include <memory>

namespace dlaf {
namespace common {

template <class U>
class IWrapper {
  public:
  IWrapper(U&& object) : object_(std::move(object)) {}

  IWrapper(IWrapper&& rhs) = default;

  ~IWrapper() {
    if (p_)
      p_->set_value(std::move(object_));
  }

  U& setPromise(hpx::promise<U>&& p) {
    if (p_)
      throw std::logic_error("setPromise has been already used on this object!");
    p_ = std::make_unique<hpx::promise<U>>(std::move(p));
    return object_;
  }

  U& operator()() {
    return object_;
  }

  const U& operator()() const {
    return object_;
  }

  private:
  U object_;
  std::unique_ptr<hpx::promise<U>> p_;
};

}
}
