# Coding Style

## Files

Each file should start with the header containing the license information provided in HEADER.

### File organization
Each class should be defined in a separate file whose name is the class name with upper case letters replaced by an underscore (`_`) and the corresponding lower case letter.
(Exception: If the upper case letter is the first letter, no underscore is prepended). E.g. the name for the file that contains the class `MyClassExample` should be `my_class_example` with the correct extension.

Files which contains collections of functions should have a name composed by lower case letter and underscores (`_`) which briefly explain the type of functions contained.
E.g. if the file contains math utilities the filename can be `util_math` with the correct extension.

Header files should be placed in `include/<libname>/`, source files in `src/`.
Tests should be placed according to their type in `test/<type>/`.

### File Names
Filenames should be all lowercase and can include numbers and underscores (`_`).
Header files use the extension `.h` and source files use the extension `.cpp`.

### Header files
All header files should be self-contained.
This means that each header can be included without defining anything special.
In particular each header file should include all the headers needed and have `#pragma once` to prevent multiple inclusion.

### Includes
Only use the syntax `#include <...>`, unless the include uses a relative path (see below).

Header files should be included with the full path based on the `include` directory.
For example, the file `{libname}/include/{libname}/foo/bar.hpp` should be included as
```c++
#include <{libname}/foo/bar.hpp>
```
Even if the included file is located in the same directory as the including file this rule should be obeyed.
Only source include files (i.e. headers not located in an `include` directory) are included using relative paths.

Never include C library headers. Use instead the corresponding C++ header.
E.g. include `cstdint` instead of `stdint.h`.

For readability it is suggested to use the standard order of includes:
* C++ library headers
* Other libraries' headers
* `pika` headers
* `dlaf` headers
* source include files
* `gtest` headers (they should be included only in the test directory)
* `dlaf_test` headers (they should be included only in the test directory)
* conditionally included files

In each group the included files should be sorted in alphabetical order,
and groups are separated by a blank line.
The clang-format style included in the repository provides the correct settings.

## Naming

The names should be as descriptive as possible.

**Note**
the following rules can be broken, to follow standard library conventions in case of similar
functionalities. E.g trait helpers ends with `_t` and `_v`.

### Type and Class Names

Type and class names should start with a capital letter and have a capital letter for each new word.
Underscores (`_`) are not allowed.

**Note**
exceptions are allowed to follow standard library conventions in case of similar
functionalities.

The `using` syntax is preferred to `typedef` for type definitions.

### Namespace Names

In general, namespace names should be one word, lowercase.
If multiple words are needed they should be lowercase and separated with an underscores (`_`).

### Variable Names

#### Variable Names

Variable names should contain only lowercase characters and underscores (`_`).
The underscore is used to separate words and is not allowed as first or last character.
Two consecutive underscores (`__`) are not allowed.

#### Class Data Members

Public data members names follows the convention of variable names.
Private class data members names follows the convention of variable names with a trailing underscore (`_`).

### Function Names

Variable names should contain only lowercase characters and underscores (`_`).
The underscore is used to separate words and is not allowed as first or last character.
Two consecutive underscores (`__`) are not allowed.

### Lambda Expressions

Named lambda expressions follow the naming convention for variables:

```c++
auto my_lambda = [](int i) { return i + 4; };
```

### Macro Names

Macro names should be all uppercase and can include underscores (`_`).
The underscore is used to separate words and is not allowed as first or last character.
Two consecutive underscores (`__`) are not allowed.

### Naming Example

```c++
#define INCREMENT_MACRO(i) ++(i)

namespace two_words {
namespace name {

int variable_name;

void function_name_multiple_words(...) {
  auto my_lambda = [](int i) { return i + 4; };
  other_statements;
}

class ClassNameMultipleWords {
public:
  ClassNameMultipleWords : member_name_(0) {}

  int get_member_name() {
    return member_name_;
  }

  void set_member_name(int new_value) {
    member_name_ = new_value;
  }

protected:
  void function_name(Type parameter1, Type2 parameter2);

private:
  int member_name_;
};
}
}
```

## Comments

### Comment Style

Prefer the `// Comment` syntax, and use the `/* ... */` syntax only when is required.

Classes and functions has to be documented with **Doxygen** (see dedicated section).

### Function Comments

Each function should have a brief description on what the function does and a more detailled documentation separated by a blank line.

The detailled documentation should contain:
* A more detailled explanation of the function
* for each function parameters a brief description of the parameter and the specification if they are input (in), output (out) or input-output parameters (in,out:),
* a description of the return value (if not included in the function description),
* preconditions
* postconditions

Example:
```c++
/// Updates foo and computes bar using in_1 .. in_5.
///
/// The function computes bar using the Foo method.
/// @param[in] in_1 the 1st input parameter
/// @param[in] in_2 the 2nd input parameter
/// @param[in] in_3 the 3rd input parameter
/// @param[in] in_4 the 4th input parameter
/// @param[in] in_5 the 5th input parameter
/// @param[in,out] foo the in,out parameter
/// @param[out] bar the output parameter
/// @return the number of iterations performed
/// @pre precondition1
/// @pre precondition2
/// ...
/// @post postcondition1
/// ...
int computeFooBar(Type in_1, const Type& in_2, Type& in_3,
                  const Type* in_4, Type* in_5, Type& foo,
                  Type& bar);
```

### Variable Comments

No comments, since the name should be self-descriptive.

### Class Comments

Every class should have a short description of what it is and what is does. If needed a detailled description can be added.
Comments for public class member functions follow the same rules as general function comments.
Comments for private members are allowed, but not mandatory.

### Implementation Comments

Tricky, complicated or important code blocks should have comments before them.

Example:
```c++
// Comment here.
statement;

// Long comment about
// next statements
statement;
other_statement;
statement3;
```

Line comments should be separated from the code by 2 spaces.
When multiple subsequent lines of code have a comment, they have to be aligned.

Example:
```c++
statement;  // Comment here.

statement;        // Comment statement
other_statement;  // Comment other statement
statement3;       // Comment statement3 (Comments are aligned)
```
If you modify a piece of code, also adapt the comments that belong to it if necessary.

## Doxygen
All classes and functions must be documented with Doxygen using the `///` style.

Doxygen commands should be specified with `@` syntax (e.g. `@param`).

General rules:

- no empty comment lines before and after the comment block,
- one space after the brief comment.

```cpp
/// Returns @c ceiling(@p num / @p den) for integer types.
///
/// @tparam IntType has to be an integer type
/// @pre @a num >= 0 and @a den >= 0
```

### Documentation & advanced languages features
Sometimes, it can be necessary to hide a complex language syntax from the documentation with the aim of
making it more clear and simple.
In all these cases it is possible to separate the code to be processed by Doxygen from the real one by
using the macro `DLAF_DOXYGEN`.

```cpp
#ifdef DLAF_DOXYGEN
code to show in doxygen
#else
code to build
#endif
```

Example:

```cpp
#ifdef DLAF_DOXYGEN
template <typename IntType>
constexpr IntType ceilDiv(const IntType num, const IntType den);
#else
template <typename IntType>
constexpr auto ceilDiv(const IntType num, const IntType den)
   -> std::enable_if_t<std::is_integral<IntType>::value, IntType> {
  return (num + den - 1) / den;
}
#endif
```

In the first part there is a simplified declaration of the function, while in the second one the real
definition. The former one will be the one processed by Doxygen, the latter the one seen by the compiler.

The two different parts should be completely independent and they should not share any part of the
declaration. For instance, in the example it would have been possible to have just one
`template <typename IntType>` as first line without repeating it, but it is discouraged.

## Formatting

Use the provided clang-format style to format `.h`, `.hpp`, `.tpp` and `.cpp` files.

### Line Length

The length of each line of your code should, in principle, be at most **105** characters.
This limit can be exceeded by few characters in special cases.

For C++ files the clang-format style included in the repository provides the correct settings.

### Scopes

Do not use scopes for formatting reason.

### Horizontal Spacing

No trailing whitespaces should be added to any line.
Use no space before a comma (`,`) and a semicolon (`;`) and add a space after them if they are not at the end of a line.

For C++ files the clang-format style included in the repository provides the correct settings.

### Parentheses

Parentheses should have no internal padding.
In general they have one space external padding, unless the following cases are met
which require no external padding:
* between two opening or two closing parentheses,
* between the function name and its arguments,
* between a closing parenthesis and a comma (`,`) or a semicolon (`;`).

For C++ files the clang-format style included in the repository provides the correct settings.

### Binary Operators

The assignment operator should always have spaces around it.
Other operators may have spaces around them, but it is not mandatory.

For C++ files the clang-format style included in the repository provides the correct settings.
**Note**
clang-format forces a space before and one after the binary operators.

### Unary Operators

Do not put any space between an unary operator and their argument.
For C++ files the clang-format style included in the repository provides the correct settings.

### Types

There is no space between the type and `*` or `&` in pointer or references.
The angle brackets of the templates should not have any external and internal padding.

Examples:
```c++
Type var;
Type* ptr = &var;
Type& ref = var;

Class1<Class2<type1>> object;
```

For C++ files the clang-format style included in the repository provides the correct settings.

### Vertical Spacing

Use empty lines when it helps to improve the readability of the code, but do not use too many.
Do not use empty lines after a brace which opens a scope,
or before a brace which closes a scope.
Each file should contain exactly one empty line at the end of the file.
**Warning**
Some editors add an empty line automatically without displaying it (e.g. vi), some do not.

### Indentation

Indentation consists of 2 spaces.
Do not use tabs in the code.

### Variable Declarations and Definitions

Do not declare multiple variables in the same declaration, especially if they are not fundamental types:

```c++
// Discouraged.
int x, y;
Object a("my-matrix"), b(10);

// Preferred way.
int x;
int y;
Object a("my-matrix");
Object b(10);
```

### Function Declarations and Definitions

Formatted according to the clang-format style included in the repository.

### Function Calls

Formatted according to the clang-format style included in the repository.

### Conditionals

Formatted according to the clang-format style included in the repository.

### Switch statement

Switch statements should always have a default case, unless an enum class is used and all the cases are covered.
Indentation according to the clang-format style included in the repository.

Example:
```c++
switch (var) {
  case 0:
    statement1;
    statement2;
    break;

  case 1:
    statement1;
    statement2;
    break;

  default:
    statement1;
    statement2;
}
```

### Loops

Formatted according to the clang-format style included in the repository.

### Preprocessor Directives

Indentation according to the clang-format style included in the repository.

### Class Format

Unless a different order is required, the members should be defined in the following order:
* using statements
* public members
* protected members
* private members

Indentation according to the clang-format style included in the repository.

### Constructor Initializer Lists

Formatted according to the clang-format style included in the repository.

## Other C++ Features

### Pre-increment and pre-decrement

Use the pre-increment (pre-decrement) operator when a variable is incremented (decremented) and the value of the expression is not used.
In particular, use the pre-increment (pre-decrement) operator for loop counters.

The post-increment and post-decrement operators create an unnecessary copy, that the compiler cannot optimize away in the case of iterators or other classes with overloaded increment and decrement operators.

### Alternative Operator Representations

Use the primary token for operators (E.g. `&&`, `||`, and `!`), alternative representations are not allowed (`and`, `or`, and `not`).

## Index naming convention

To have an uniform usage of names for indices and sizes in the library, please follow the following conventions:
- Use `m`, `n` and `k` for sizes and `i`, `j` and `l` for indices,
- in general, `i` and `m` refers to rows (exceptions square sizes and diagonal),
- in general, `j` and `n` refers to columns (exceptions square sizes and diagonal).

**Note**
if the use of the row or column index/size is determined by a template or a function parameter `i`, `m` should be used for the "parallel" coordinate and `j`, `n` for the orthogonal one.

2D Indices:
- Use the two indices to name the 2D object. E.g. `ij`. In this way extracting a single index with `SizeType i = ij.row();` is intuitive.
- The use of `id`, `idx` and similar names is discouraged.

2D Size:
- Name the object to identify what it refers to. E.g. `size_a` represents the number of global tiles of the matrix A.

Distinction of different type of 1D and 2D indices and 1D sizes:
- GlobalElementIndex and GlobalElementSize: `_el` suffix, e.g. `j_el`.
- LocalElementIndex and LocalElementSize: `_el_lc` suffix, e.g. `i_el_lc`
- GlobalTileIndex and GlobalTileSize: no suffix, e.g. `m`
- LocalTileIndex and LocalTileSize: `_lc` suffix, e.g. `n_lc`
- TileElementIndex and TileElementSize: `_el_tl`, e.g. `i_el_tl`.

Distinction of different type of 2D sizes:
- GlobalElementSize, LocalElementSize, GlobalTileSize, LocalTileSize: as above.
- TileElementSize: use `tile_size` as base of the name, e.g. `tile_size_a` to name the size of a tile of the matrix A.
