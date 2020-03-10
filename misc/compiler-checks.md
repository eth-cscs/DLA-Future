# Compiler Checks

## Supported Compilers

At the moment we are supporting the following compilers:

- GCC
- Clang (and AppleClang)

## Warnings

Every compiler has its own set of warning checks. Here we list them more relevant differences among them.

### -Wconversion

In GCC `-Wconversion` is more strict than the same one on Clang. In fact, on GCC it triggers both on downcasts (e.g `int16_t` > `int8_t`) and on float to int conversions, while in Clang float to int conversions are not involved.

Clang 11 provides `-Wimplicit-int-float-conversion` (added to the `-Wconversion` gruop), which specifically addresses the aforementioned case.

### -Wdangling-else

In Clang `-Wdangling-else` is not enable by default, while in GCC it is.
