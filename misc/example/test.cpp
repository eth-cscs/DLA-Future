#include "m.h"

#include <iostream>
#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/runtime.hpp>
#include <pika/thread.hpp>

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
  pika::threads::scheduled_executor HP =
      pika::threads::executors::pool_executor("default", pika::threads::thread_priority_high);

  // setup matrices
  using Type = Tile<int>;

  std::array<int, 4> a = {0, 1, 2, 3};
  std::array<int, 4> b = {0, 1, 2, 3};
  std::array<pika::future<Type>, 4> fa = {pika::make_ready_future<Type>(&a[0]),
                                         pika::make_ready_future<Type>(&a[1]),
                                         pika::make_ready_future<Type>(&a[2]),
                                         pika::make_ready_future<Type>(&a[3])};

  std::array<pika::future<Type>, 4> fb = {pika::make_ready_future<Type>(&b[0]),
                                         pika::make_ready_future<Type>(&b[1]),
                                         pika::make_ready_future<Type>(&b[2]),
                                         pika::make_ready_future<Type>(&b[3])};

  Matrix<int> ma(std::move(fa));
  Matrix<int> mb(std::move(fb));
  // end of matrices setup

  // execute some operation on matrix B asynchronously.
  MatrixRW<int> mb1 = mb.block();
  pika::async([mb1 = std::move(mb1)]() mutable { pika::dataflow(pika::unwrapping(work0), mb1(0), 1); });

  // execute some operation on matrix A asynchronously. B is constant (read-only).
  MatrixRW<int> ma1 = ma.block();
  MatrixRead<int> mb2 = mb.block_read();
  pika::async([ma1 = std::move(ma1), mb2 = std::move(mb2), &HP]() mutable {
    for (std::size_t i = 1; i < 4; ++i) {
      pika::dataflow(pika::unwrapping(work2<>), ma1(0), ma1.read(i), 100 + i);
    }
    { pika::dataflow(pika::unwrapping(work2<>), ma1(2), ma1.read(3), 200); }
    for (std::size_t i = 0; i < 4; ++i) {
      pika::dataflow(pika::unwrapping(work2<>), ma1(i), mb2.read(i), 300 + i);
      pika::dataflow(HP, pika::unwrapping(work), ma1(1), i * 100, 400 + i);
    }
  });

  // execute some work on A and B.
  for (std::size_t i = 0; i < 4; ++i) {
    pika::dataflow(pika::unwrapping(work), mb(i), -1, 1000 + i);
    pika::dataflow(pika::unwrapping(work2<false>), ma(i), mb.read(i), 2000 + i);
    pika::dataflow(pika::unwrapping(work), ma(3), i * 10000, 3000 + i);
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
  pika::start(nullptr, argc, argv);
  pika::runtime* rt = pika::get_runtime_ptr();
  pika::util::yield_while([rt]() { return rt->get_state() < pika::state_running; });

  pika::threads::run_as_pika_thread(foo);

  pika::apply([]() { pika::finalize(); });
  return pika::stop();
}
