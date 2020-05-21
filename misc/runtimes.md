Design rationale:

There are a few alternative HPX/CUDA implementation options:

1) The approach taken here inserts an event just after a CUBLAS function is called and uses it to
   track when the function has finished. Each task calls `hpx::util::yield_while()` and
   `cudaEventQuery()` to infrom HPX's scheduler when it's is ready.

2) The current implementation in HPX inserts a callback function on a stream using
   `cudaStreamAddCallback()`. The callback registers the OS thread on which the stream is running
   with HPX and sets the data associated with a custom future type to indicate that a preceding CUDA
   function has completed.

  There are a few issues with this approach:

   - `cudaStreamAddCallback()` is deprecated in recent CUDA versions and is now superseded by
     `cudaLaunchHostFunc()`.
   - HPX throws an error if an external thread is registered multiple times, which happens as a
     result of the callback being called multiple times. To make this work, the error is currently
     ignored, which seems a little hacky.
   - The design is based on `targets` instead of `executors`.

3) [NOT IMPLEMENTED] This approach is similar to how MPI Futures are implemented. `cudaEvent_t` are
   stored in a global array and polled for completion whenever a task yields or completes on a pool
   that has the polling function registered.

It is unclear at the moment which approach is the best. According to John 1) and 3) ought to have
very similar performance. The two approaches are also available to MPI, performance comparisons
there did not produce a clear winner which appears to confirm John's statement.

Most CUDA functions are thread safe, that includes Streams and CUBLAS handles. The executor accepts a
`pool_executor` which specifies the host threads on which the CUBLAS calls are scheduled. There is a
price to pay for potentially sharing the CUBLAS handles between multiple threads:

 - `cudaSetDevice()` needs to precede the CUBLAS call to make it's handle valid on that thread
 - a mutex is needed to ensure the event is scheduled after the call and there is nothing in-between

We can eliminate both of these if we can dedicate a host thread to a particular device and only
execute CUBLAS / CUDA calls there. There are two approaches to achieve that with HPX:

a) separate pools with a single host threads per device
b) force bind tasks to OS threads
c) TLS (thread local storage)-like approach (TODO)

Apporach b) can work If either the static scheduler or John's scheduler (shared-priority-scheduler)
is used. Otherwise a hpx task is not guaranteed to actually run on the OS thread where it was
scheduled (it can get stolen). That is the case even if a schedule hint with the desired thread
number is provided. Note also that according to John, binding tasks within his scheduler is still
rather experimental.

Approach a) is the way to go but there are a few caveats. The following about HPX is good to know:

- Two HPX thread pools can't share OS threads
- The number OS threads used by HPX is fixed after it's initialized
- By default HPX doesn't allow oversubscribing cores with OS threads

Since a separate CUDA/CUBLAS host thread for a device will likely do little compared to other HPX
threads, we wouldn't want to dedicate a core to it. To overcome that, we have to enable thread
oversubscription in HPX and allow to schedule more than one HPX OS thread per core. We can do that via
the `hpx::resource::mode_allow_oversubscription` at initialization within the resource partitioner.
