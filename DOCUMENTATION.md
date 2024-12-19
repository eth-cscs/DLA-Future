# DLA Future Documentation

## API documentation

- [Documentation of `master` branch](https://eth-cscs.github.io/DLA-Future/master/)
- [Documentation of `v0.7.2`](https://eth-cscs.github.io/DLA-Future/v0.7.2/)
- [Documentation of `v0.7.1`](https://eth-cscs.github.io/DLA-Future/v0.7.1/)
- [Documentation of `v0.7.0`](https://eth-cscs.github.io/DLA-Future/v0.7.0/)
- [Documentation of `v0.6.0`](https://eth-cscs.github.io/DLA-Future/v0.6.0/)
- [Documentation of `v0.5.0`](https://eth-cscs.github.io/DLA-Future/v0.5.0/)
- [Documentation of `v0.4.1`](https://eth-cscs.github.io/DLA-Future/v0.4.1/)
- [Documentation of `v0.4.0`](https://eth-cscs.github.io/DLA-Future/v0.4.0/)
- [Documentation of `v0.3.1`](https://eth-cscs.github.io/DLA-Future/v0.3.1/)
- [Documentation of `v0.3.0`](https://eth-cscs.github.io/DLA-Future/v0.3.0/)
- [Documentation of `v0.2.1`](https://eth-cscs.github.io/DLA-Future/v0.2.1/)
- [Documentation of `v0.2.0`](https://eth-cscs.github.io/DLA-Future/v0.2.0/)
- [Documentation of `v0.1.0`](https://eth-cscs.github.io/DLA-Future/v0.1.0/)

### How to generate the documentation

Note: [Doxygen](https://www.doxygen.nl/) is required to generate the documentation.

The documentation can be built together with the project by enabling its generation with the flag `DLAF_BUILD_DOC=on` and then use the `doc` target to eventually generate it.

```bash
# from the build folder, if you have already configured the CMake project
cmake -DDLAF_BUILD_DOC=on .
make doc
```

Alternatively, the documentation can be generated independently by using `doc/Doxyfile.in`, which is a template configuration file in which you have to replace the text `${DLAF_SOURCE_DIR}` with the root folder of DLAF containing the source code (e.g. where you cloned this repository).

As a shortcut for this process a `doc/Makefile` is available, which automatically performs the substitution and then generates the documentation.

```
cd doc
make doc
```
