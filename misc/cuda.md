# CUDA/cuBLAS executors

The current cuBLAS executor implementation lives in DLA-Future. This will be
upstreamed to HPX once stable.

There are three main design considerations for the cuBLAS executors:

1. The synchronization mechanism.
2. Tile lifetime management.
3. Stream and handle management

## Synchronization

The executor currently relies on an experimental customization point in the HPX
dataflow implementation to improve integration with hpx::dataflow. It uses
HPX's new internal polling mechanism to check for completion of kernels. After
submitting a kernel for execution on the GPU, the executor registers an event
with the scheduler, which then polls for completion between executing tasks.

There are two alternative approaches to synchronization. The first is to use
CUDA callbacks (currently `cudaStreamAddCallback`, in the future
`cudaLaunchHostFunc`) to trigger completion of futures. Benchmarks show that
this is likely slightly or a lot slower than using events. The second
alternative is to poll "manually" in a user-created task using `yield_while`.
The benefit of this approach is that it does not require scheduler integration,
but may lead to higher overheads when waiting for many kernels, as each kernel
is polled in independent tasks. However, initial benchmarks show that this
approach performs about the same as the scheduler-based polling.

## Tile lifetime

When launching CPU BLAS functions in tasks, tiles live naturally for the
duration of the BLAS operation. cuBLAS operations are asynchronous and naively
launching cuBLAS operations without ensuring that tiles live for the duration
of the operation releases the tiles for use by subsequent tasks too early.
There are two approaches that can be used here. The first is to simply make
cuBLAS wrapper functions blocking until the operation is complete. The second
is to return the tiles from the wrapper functions to ensure that they live
until the future representing the operation is released. The initial
implementation uses the second approach to avoid unnecessary yielding of HPX
tasks. However, the overheads from either approach are unlikely to be high
compared to CUDA latencies.

## Stream and handle management

The current implementation uses a pool of CUDA streams and a single cuBLAS
handle. Access to these are locked with a single lock to ensure that event
registration is done in the order that tasks are submitted. A potential
enhancement is to use thread-local streams (and possibly cuBLAS handles as
well) to avoid locking.
