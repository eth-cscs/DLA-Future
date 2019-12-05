#pragma once

#include <hpx/lcos/promise.hpp>

namespace dlaf {
namespace common {

template <class U>
class IWrapper {
  public:
  IWrapper(U&& object) : object_(std::move(object)), is_valid_(false) {}

  IWrapper(IWrapper&& rhs) : object_(std::move(rhs.object_)), is_valid_(std::move(rhs.is_valid_)), promise_(std::move(rhs.promise_)) {
    rhs.is_valid_ = false;
  }

  ~IWrapper() {
    if (is_valid_)
      promise_.set_value(std::move(object_));
  }

  U& setPromise(hpx::promise<U>&& p) {
    if (is_valid_)
      throw std::logic_error("setPromise has been already used on this object!");
    is_valid_ = true;
    promise_ = std::move(p);
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
  bool is_valid_;
  hpx::promise<U> promise_;
};

}
}
