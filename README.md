# Distributed Linear Algebra with Futures.

## Dependencies

- MPI
- [HPX](https://github.com/STEllAR-GROUP/hpx)
- [blaspp](https://bitbucket.org/icl/blaspp/src/default/)
- [lapackpp](https://bitbucket.org/icl/lapackpp/src/default/)
- Intel MKL or other LAPACK implementation
- [cuBLAS](https://developer.nvidia.com/cublas) (optional)
- [GoogleTest](https://github.com/google/googletest) (optional; bundled) - unit testing
- Doxygen (optional) - documentation

## How to use the library

Using DLAF in a CMake project is extremly easy!

Let's use the variable `$DLAF_ROOT` for referring to the install path of DLAF.

Configure your project with:

`cmake -DDLAF_DIR="$DLAF_ROOT/lib/cmake" ..`

Then, it is just as simple as:

```
find_package(DLAF)

# ...

target_link_libraries(<your_target> PRIVATE DLAF)
```
