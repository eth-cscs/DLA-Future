# Synchronization mechanism.

To simplify the dependency tracking during the generation of the task graph we introduced
a mechanism which generates automatically the following dependencies:
- Subsequent tasks that modify (write) values of a tile are ordered, therefore each write
  task depends on the previous write task.
- Subsequent tasks that only read values a tile can be executed concurrently,
  however they all depends on the previous write task and the next write task depends on
  the completion of all the read tasks.

## Matrix and Matrix Tiles basics

The Matrix object (`dlaf::matrix::Matrix`) is the object which manage the setup of
the correct dependencies of the tasks which involve any of its tiles.

Matrix Tiles (`dlaf::matrix::Tile`) can be accessed using two Matrix methods.
- operator() returns a `hpx::future` of a tile containing the requested tile,
- read() returns a copy of a `hpx::shared_future` of a tile with constant elements representing
  the tile (allows only read operations on tile elements).

Subsequent calls to read() with the same tile index return a copy of the same `shared_future`
if no operator() was invoked.

Both the Matrix and the Tile objects are not thread safe. The future mechanism ensures that
each tile is either accessed in read-write mode by only one task, or in read-only mode by more tasks.
On the other hand the user has to guarantee that a Matrix object is used only by one task at a time.
For asynchronous scheduling [matrix views](#matrix-views) has to be used.

### Basic Examples

(Note: this examples are used to illustrate how dependencies are determined,
however they cannot be compiled since they are simplified to improve readability.
(e.g. constructors and template parameters are omitted)).

- Example 1: Write dependencies.
```
  Matrix m;

  hpx::dataflow(Task1, m({0, 0})));
  hpx::dataflow(Task2, m({0, 0})));  // Depends on Task1.
  hpx::dataflow(Task3, m({0, 1})));  // Different tile. No dependency on Task1 or Task2.

// Resulting dependency graph:
// Task1 - Task2
//
// Task3
```
- Example 2: Read dependencies.
```
  Matrix m;

  hpx::dataflow(Task1, m({0, 0})));
  hpx::dataflow(Task2, m({0, 1})));  // Different tile.
  hpx::dataflow(Task3, m.read({0, 0}), m({1, 1})));  // Depends on Task1.
  hpx::dataflow(Task4, m.read({0, 0}), m({0, 1})));  // Depends on Task1 and Task2. No dependency on Task3 (both only read Tile {0, 0})
  hpx::dataflow(Task5, m({0, 0})));  // Depends on Task3 and Task4.
  hpx::dataflow(Task6, m.read({0, 0})));  // Depends on Task5.

// Resulting dependency graph:
// Task1 - Task3 - Task 5 - Task6
//       \       /
// Task2 - Task4
```

### Exception handling

If a task throws an exception during its execution the future returned by the asynchronous call
contains the given exception.  Moreover any tile that was supposed to be written by the Task will
be invalidated and any subsequent call to `operator()` and `read()` will return a future containing
an exception. Note that because of limitations we cannot propagate the correct exception type and
message to tiles. Therefore they will contain a generic exception.

## Matrix Views

Matrix Views (`dlaf::matrix::MatrixView`) are objects which allow to schedule the DAG of an
algorithm in a different task.

It can be created from a Matrix or from another MatrixView which will be indicated as "parent matrix".
Both constant and non-constant views are available. The main difference is that `operator()`
is not allowed for constant matrices as only read-only tasks can be performed on its tiles.

When a matrix view is created each subsequent call of `operator()` and `read()` on the parent matrix
will return futures which depend on the tasks scheduled with the futures obtained by the matrix view.
It can be noted that it is possible to transfer the scheduling on only a part of the tiles.
Currently it can be specified with the `uplo` parameter if:
- all the tiles or
- only the tiles which contain an element of the diagonal or the upper (lower) triangular

has to be transferred to the new view.

In this way the DAG scheduling can be performed concurrently on the parent matrix and the matrix view,
but the correct dependencies are guarenteed. An important fact to be noted is that the view needs to be
able to notify that no other tasks will be scheduled to enable the next tasks created from the parent matrix.
The `MatrixView` API contains 3 ways to accomplish that:
- the destructor notifies that no new tasks will be scheduled on any of the tile,
- `done(tile_index)` notifies that no new task will be scheduled on the tile with the given index
(After this call any use of `operator()` or `read()` with the same tile index will return an error.),
- `doneWrite(tile_index)` notifies that no new read-write task will be scheduled on the tile with the given index
(After this call any use of `operator()` with the same tile index will return an error.).

It has to be noted that read-only task of the parent matrix and of the View can be performed at the same time
if no read-write tasks has been scheduled in between (see [examples](#view-examples)).

### Examples

(Note: this examples are used to illustrate how dependencies are determined,
however they cannot be compiled since they are simplified to improve readability.
(e.g. constructors and template parameters are omitted)).

- Example 1: Basic view usage.
```
  Matrix m;

  hpx::dataflow(Task1, m({0, 0})));
  hpx::dataflow(Task2, m({0, 0})));  // Depends on Task1.
  hpx::dataflow(Task3, m({0, 1})));  // Different tile. No dependency on Task1 or Task2.

  {
    MatrixView mv(UpLo::General, m);
    hpx::dataflow(Task4, mv({0, 0})));  // Depends on Task2.
    hpx::dataflow(Task5, mv({0, 1})));  // Depends on Task3.
  }

  hpx::dataflow(Task6, m({0, 0})));  // Depends on Task4.
  hpx::dataflow(Task7, m({0, 1})));  // Depends on Task5.

// Resulting dependency graph:
// Task1 - Task2 - Task4 ~ Task6
//
// Task3 - Task5 ~ Task7
//
// Note the that the ~ dependencies include the call to the mv destructor.
```
- Example 2: Basic Usage with `done()`.
```
  Matrix m;

  hpx::dataflow(Task1, m({0, 0})));
  hpx::dataflow(Task2, m({0, 0})));  // Depends on Task1.
  hpx::dataflow(Task3, m({0, 1})));  // Different tile. No dependency on Task1 or Task2.

  MatrixView mv(UpLo::General, m);

  hpx::dataflow(Task4, m({0, 0})));  // Depends on done notification and Task6
  hpx::dataflow(Task5, m({0, 1})));  // Depends on done notification and Task7
  hpx::dataflow(Task6, mv({0, 0})));  // Depends on Task2.
  hpx::dataflow(Task7, mv({0, 1})));  // Depends on Task3.

  mv.done({0, 0});
  mv.done({0, 1});

// Resulting dependency graph:
// Task1 - Task2 - Task6 ~ Task4
//
// Task3 - Task7 ~ Task5
//
// Note the that the ~ dependencies include the dependency on the call to done().
```
- Example 3: Read cuncurrency.
```
  Matrix m;

  hpx::dataflow(Task1, m({0, 0})));
  hpx::dataflow(Task2, m.read({0, 0})));  // Depends on Task1

  MatrixView mv(UpLo::General, m);

  hpx::dataflow(Task3, m.read({0, 0})));  // Depends on doneWrite notification and Task5
  hpx::dataflow(Task4, mv.read({0, 0})));  // Depends on Task2.
  hpx::dataflow(Task5, mv({0, 0})));  // Depends on Task2 and Task4.
  hpx::dataflow(Task6, mv.read({0, 0})));  // Depends on Task5.

  mv.doneWrite({0, 0});
  hpx::dataflow(Task7, m.read({0, 0})));  // Depends on doneWrite notification and Task5
  hpx::dataflow(Task8, m({0, 0})));  // Depends on done notification and Task3,6,7
  mv.done({0, 0});

// Resulting dependency graph:
// Task1 - Task2 - Task5 -  Task6 -= Task8
//       \ Task4 /       \~ Task3 /
//                       \~ Task7 /
//
// Note the that the \~ dependencies include the dependency on the call to doneWrite().
// Note the that the -= dependencies include the dependency on the call to done().
```

## Matrix and Matrix Tiles advanced: Tile dependencies mechanism

The dependency are managed with a promise-future mechanism.

When the destructor of a Tile obtained with `operator()` or `read()` is called, the future
of the next future is set using the promise which was assigned by the Matrix object.
Note that the destructor of the Tile contained in the shared future is only called when all
instances of the shared futures go out of scope, therefore,
to avoid deadlocks the scope of tile shared futures needs extra care.
Note that returning matrix tiles from tasks may be a source of deadlocks,
therefore a carefull analysis of dependencies to avoid problems.

### Examples advanced use

- Example 1: Returning Tiles.
```
Tile&& Task1(hpx::future<Tile>&& future) {
  auto tile = future.get();
  // Do some work

  return std::move(tile);
}

  Matrix m;

  auto future1 = hpx::dataflow(Task1, m({0, 0})));
  hpx::dataflow(Task2, m.read({0, 0}));  // Depends on Task3 (The Tile used in Task1 is still available in future1).

  hpx::dataflow(Task3, future1);  // Depends on Task1.

// Resulting dependency graph:
// Task1 - Task3 - Task2
```
- Example 2: Deadlock returning Tiles.
```
Tile&& Task1(hpx::future<Tile>&& future) {
  auto tile = future.get();
  // Do some work

  return std::move(tile);
}

  Matrix m;

  auto future1 = hpx::dataflow(Task1, m({0, 0})));
  auto future2 = hpx::dataflow(Task2, m.read({0, 0}));  // Depends on Task3 (The Tile used in Task1 is still available in future1).

  future2.get();  // DEADLOCK! Task2 is not ready yet, and cannot get ready because the destructor of the Tile in future1 will not be called.

  hpx::dataflow(Task3, future1);  // Depends on Task1.
```
- Example 3: Deadlock returning Tiles (2).
```
Tile&& Task1(hpx::future<Tile>&& future) {
  auto tile = future.get();
  // Do some work

  return std::move(tile);
}

  Matrix m;

  auto future1 = hpx::dataflow(Task1, m({0, 0})));
  hpx::dataflow(Task2, future1, m.read({0, 0}));
  // DEADLOCK:
  // This task depends on Task1 (via future1), and on the destruction of the Tile in future1 (read()).
  // The scope of the Tile in future1 is extended to the end of Task2, and therefore the execution
  // of Task2, and therefore of the destructor of future1 tile, depends on the execution of itself.
```
- Example 4: Deadlock shared future scope.
```
  Matrix m;

  auto future1 = hpx::dataflow(Task1, m({0, 0})));
  auto shared_future = matrix.read({0, 0});
  hpx::dataflow(Task2, shared_future);  // Depends on Task1.
  hpx::dataflow(Task3, m.read({0, 0}));  // Depends on Task1.

  auto future4 = hpx::dataflow(Task4, m({0, 0}));  // Depends on Task2 and Task3.

  future4.get();  // DEADLOCK! Task4 is not ready yet, and cannot get ready because the destructor of the Tile in shared_future will not be called.
```

## Implementation details

To explain how the dependencies mechanism works we build the mechanism from simple cases adding
features one by one.

The simplest case includes constant matrices with read-only tiles. In this case only a shared future
for each tile is needed which is returned by the `read()` operator.

The next case to be considered is a matrix for which read-only access is not be needed.
For this specific case a future for each tile is needed. This future is an "internal" future
since the promise of the tile is not set yet.
When the `operator()` is called a promise in injected into the future with a synchronous continuation
(i.e. a continuation which is executed as soon as the future gets ready) and the future of the promise
is kept by the matrix.

In the general case when the `read()` operator has to be available as well a future and a
shared future are needed for each tile.
The usage of the future is the same as in the previous case, with an extra operation: A call to `operator()`
sets invalidates the shared future setting it to a default constructed object.
On the other hand the shared future is the main change to handle the read-only case.
A call to `read()` first checks if the shared future is valid. If it is not it means that
a read-write operation was scheduled before and the shared future has to be constructed.
The mechanism is similar to the one used in `operator()` with the only difference that
the future contains a tile with constant element ant it is transformed in a shared future.
A copy of the shared future is stored in the matrix for successive call to `read()` which simply return
the stored object.

Extra care has to be used for correct exception handling.
The tile destructor sets the promise value if it was called during normal execution, while it sets
an exception when it is called after an exception has been thrown.
If the future f contains an exception the call to `get()` in the continuation will rethrow the exception,
the promise will be destructed and the new future stored in the matrix will have an invalid shared state.
Therefore the continuations in `operator()` and `read()` have to handle this case.

Constant matrix views can be handled in a simple way. On construction it is only needed to call the `read()`
method of the parent matrix to get a copy of the shared future that is saved in the view and returned
each time the view is used. To release the view the `done()` method just clears the shared future.

For non constant matrix view the mechanism works in this way. On construction the futures of the
parent matrix are moved in the view and replaced by futures of the promises (two for each tile) also
stored in the view.
When `doneWrite()` is called the view creates its shared future and duplicates it to set the shared future of the
parent matrix too. Each of the two shared future has a corresponding future which gets ready when the tile included
is destructed, therefore a synchrounous dataflow is needed to set the future of the parent matrix
only when both shared futures are destructed.
On the other hand the `done()` method invokes `doneWrite()` if the tile is still in read-write mode,
and then it clears the shared future of the view, to allow the future in the parent matrix to get ready.

Details about the matrix and the views are illustrated in Figure 1.

