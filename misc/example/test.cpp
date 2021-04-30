#include "m.h"

#include <hpx/hpx_start.hpp>
#include <hpx/runtime/threads/run_as_hpx_thread.hpp>
#include <iostream>
#include "hpx/include/parallel_executors.hpp"

// Subtract 1 to tile value.
void work0(const Tile<int>&& i, int index) {
  std::cout << out("start", i(), -1, index);
  sleep(5);
  std::cout << out("stop", i(), -1, index);
  --i();
}

// Sum j to tile value.
void work(const Tile<int>&& i, int j, int index) {
  std::cout << out("start", i(), j, index);
  sleep(1);
  std::cout << out("stop", i(), j, index);
  i() += j;
}

// Sum (plus = true)/Subtract the value of the tile j to the value of tile i.
template <bool plus = true>
void work2(const Tile<int>&& i, const Tile<const int>& j, int index) {
  std::cout << out("start2", i(), j(), index);
  sleep(1);
  std::cout << out("stop2", i(), j(), index);
  if (plus)
    i() += j();
  else
    i() -= j();
}

void foo() {
  hpx::threads::scheduled_executor HP =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // setup matrices
  using Type = Tile<int>;

  std::array<int, 4> a = {0, 1, 2, 3};
  std::array<int, 4> b = {0, 1, 2, 3};
  std::array<hpx::future<Type>, 4> fa = {hpx::make_ready_future<Type>(&a[0]),
                                         hpx::make_ready_future<Type>(&a[1]),
                                         hpx::make_ready_future<Type>(&a[2]),
                                         hpx::make_ready_future<Type>(&a[3])};

  std::array<hpx::future<Type>, 4> fb = {hpx::make_ready_future<Type>(&b[0]),
                                         hpx::make_ready_future<Type>(&b[1]),
                                         hpx::make_ready_future<Type>(&b[2]),
                                         hpx::make_ready_future<Type>(&b[3])};

  Matrix<int> ma(std::move(fa));
  Matrix<int> mb(std::move(fb));
  // end of matrices setup

  // execute some operation on matrix B asynchronously.
  MatrixRW<int> mb1 = mb.block();
  hpx::async(
      [mb1 = std::move(mb1)]() mutable { hpx::dataflow(hpx::util::unwrapping(work0), mb1(0), 1); });

  // execute some operation on matrix A asynchronously. B is constant (read-only).
  MatrixRW<int> ma1 = ma.block();
  MatrixRead<int> mb2 = mb.block_read();
  hpx::async([ma1 = std::move(ma1), mb2 = std::move(mb2), &HP]() mutable {
    for (std::size_t i = 1; i < 4; ++i) {
      hpx::dataflow(hpx::util::unwrapping(work2<>), ma1(0), ma1.read(i), 100 + i);
    }
    { hpx::dataflow(hpx::util::unwrapping(work2<>), ma1(2), ma1.read(3), 200); }
    for (std::size_t i = 0; i < 4; ++i) {
      hpx::dataflow(hpx::util::unwrapping(work2<>), ma1(i), mb2.read(i), 300 + i);
      hpx::dataflow(HP, hpx::util::unwrapping(work), ma1(1), i * 100, 400 + i);
    }
  });

  // execute some work on A and B.
  for (std::size_t i = 0; i < 4; ++i) {
    hpx::dataflow(hpx::util::unwrapping(work), mb(i), -1, 1000 + i);
    hpx::dataflow(hpx::util::unwrapping(work2<false>), ma(i), mb.read(i), 2000 + i);
    hpx::dataflow(hpx::util::unwrapping(work), ma(3), i * 10000, 3000 + i);
  }

  // print the value of A and B.
  for (std::size_t i = 0; i < 4; ++i) {
    int ai = ma(i).get()();
    std::cout << "Result A(" << i << ") = " << ai << std::endl;
    int bi = mb(i).get()();
    std::cout << "Result B(" << i << ") = " << bi << std::endl;
  }
  std::cout << "End" << std::endl;
}

int main(int argc, char** argv) {
  hpx::start(nullptr, argc, argv);
  hpx::runtime* rt = hpx::get_runtime_ptr();
  hpx::util::yield_while([rt]() { return rt->get_state() < hpx::state_running; });

  hpx::threads::run_as_hpx_thread(foo);

  hpx::apply([]() { hpx::finalize(); });
  return hpx::stop();
}
