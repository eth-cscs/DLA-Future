#include <gtest/gtest.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/matrix.h"

#include "dlaf_test/util_tile.h"

using namespace dlaf;

template <class T>
struct Pipeline {
public:
  template <class U>
  class Wrapper {
    friend class Pipeline<U>;

    Wrapper(U& object) : object_(object) {}

  public:
    Wrapper(Wrapper&& rhs) : object_(rhs.object_) {
      promise_ = std::move(rhs.promise_);
    }

    ~Wrapper() {
      if (promise_)
        promise_->set_value(Wrapper<U>(object_));
    }

    U& get_value() {
      return object_;
    }

  private:
    Wrapper<U>& set_promise(hpx::promise<Wrapper<U>>&& next_promise) {
      assert(!promise_);
      promise_ = std::make_unique<hpx::promise<Wrapper<U>>>(std::move(next_promise));
      return *this;
    }

    U& object_;
    std::unique_ptr<hpx::promise<Wrapper<U>>> promise_;
  };

  Pipeline(T object) : object_(object) {
    future_ = hpx::make_ready_future(std::move(Wrapper<T>(object_)));
  }

  ~Pipeline() {
    if (future_.valid())
      future_.get();
  }

  hpx::future<Wrapper<T>> operator()() {
    auto before_last = std::move(future_);

    hpx::promise<Wrapper<T>> promise;
    future_ = promise.get_future();

    return before_last.then(hpx::launch::async,
                            [p = std::move(promise)](hpx::future<Wrapper<T>>&& fut) mutable {
                              return std::move(fut.get().set_promise(std::move(p)));
                            });
  }

private:
  T object_;
  hpx::future<Wrapper<T>> future_;
};

TEST(Basic, Simulation) {
  static_assert(NUM_MPI_RANKS == 2, "This test requires exactly 2 MPI ranks");
  using TypeParam = float;

  matrix::LayoutInfo layout = matrix::colMajorLayout({10, 10}, {2, 2}, 10);
  memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
  TypeParam* p = mem();
  auto matrix = createMatrixFromColMajor<Device::CPU>({10, 10}, {2, 2}, 10, p, mem.size());

  dlaf_test::tile_test::set(matrix({0, 0}).get(), 0);

  dlaf::comm::Communicator world(MPI_COMM_WORLD);

  Pipeline<dlaf::comm::Communicator> serial_comm(world);

  if (world.rank() == 0) {
    matrix({0, 0}).then(hpx::util::unwrapping([](auto&& tile) {
      CHECK_TILE_EQ(0, tile);
      dlaf_test::tile_test::set(tile, 13);
      CHECK_TILE_EQ(13, tile);
      return std::move(tile);
    }));

    hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                    dlaf::comm::sync::broadcast::send(comm_wrapper.get_value(), tile);
                    CHECK_TILE_EQ(13, tile);
                  }),
                  matrix.read({0, 0}), serial_comm());
  }
  else {
    auto last = serial_comm().then(hpx::util::unwrapping([](auto&& comm_wrapper) mutable {
      matrix::LayoutInfo layout = matrix::colMajorLayout({2, 2}, {2, 2}, 2);

      dlaf::Tile<TypeParam, dlaf::Device::CPU> workspace({2, 2}, {layout.minMemSize()}, 2);
      dlaf::comm::sync::broadcast::receive_from(0, comm_wrapper.get_value(), workspace);
      return std::move(workspace);
    }));

    last.then(hpx::util::unwrapping([](auto&& tile) { CHECK_TILE_EQ(13, tile); }));
  }
}
