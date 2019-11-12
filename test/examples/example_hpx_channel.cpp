#include <hpx/hpx_init.hpp>
#include <hpx/lcos/local/channel.hpp>

template <typename T, std::size_t PoolSize>
class Pool {
  template <class U>
  class Wrapper {
    friend class Pool<U, PoolSize>;

    Wrapper(hpx::lcos::local::channel<U>* channel, U&& object)
        : channel_(channel), object_(std::move(object)) {}

  public:
    Wrapper(Wrapper&& rhs) : channel_(rhs.channel_), object_(std::move(rhs.object_)) {
      rhs.channel_ = nullptr;
    }

    ~Wrapper() {
      if (channel_)
        channel_->set(std::move(object_));
    }

    U& get_value() {
      return object_;
    }

  private:
    U object_;
    hpx::lcos::local::channel<U>* channel_;
  };

public:
  Pool() {
    for (int i = 0; i < PoolSize; ++i)
      channel_.set(T{});
  }

  ~Pool() {
    channel_.close(/*true*/);  // TODO check what force_delete does
    for (int i = 0; i < PoolSize; ++i)
      channel_.get().get();
  }

  hpx::future<Wrapper<T>> get() {
    return channel_.get().then(hpx::launch::sync,
        hpx::util::unwrapping(std::bind(&Pool::make_wrapper, this, std::placeholders::_1)));
  }

private:
  Wrapper<T> make_wrapper(T&& object) {
    return Wrapper<T>{&channel_, std::move(object)};
  }

  hpx::lcos::local::channel<T> channel_;
};

int hpx_main(int argc, char* argv[]) {
  Pool<int, 2> pool;

  auto step1 = pool.get();
  auto step2 = pool.get();
  auto step3 = pool.get();

  step1.get().get_value() = 13;

  std::cout << step2.get().get_value() << std::endl;
  std::cout << step3.get().get_value() << std::endl;

  return hpx::finalize();
}

int main(int argc, char* argv[]) {
  return hpx::init(argc, argv);
}
