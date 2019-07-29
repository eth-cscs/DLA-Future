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
Header files use the extension `.h`, template implementation files `.ipp` and source files use the extension `.cpp`.

### Header files
All header files should be self-contained.
This means that each header can be included without defining anything special.
In particular each header file should include all the headers needed and have `#pragma once` to prevent multiple inclusion.

### Template implementation files
Template implementation files are not self-contained.
They can only be included in specific parts of the code.

In general the definition of template classes, functions, ... should be included in the header files.

All header files should be self-contained.
This means that each header can be included without defining anything special.
In particular each header file should include all the headers needed and have `#pragma once` to prevent multiple inclusion.

### Includes
Header files should be included with the full path based on the `include` directory.
For example, the file `<libname>/include/<libname>/foo/bar.hpp` should be included as
```c++
#include "<libname>/foo/bar.hpp"
```
Even if the included file is located in the same directory as the including file this rule should be obeyed.

For readability it is suggested to use the standard order of includes:

* Related header (i.e. `filename.h` for `filename.cpp` or `test_filename.cpp`)
* C library headers
* C++ library headers
* Other libraries' headers
* <libname> headers

In each section the included files should be sorted in alphabetical order.

## Naming

The names should be as descriptive as possible.

### Type and Class Names

Type and class names should start with a capital letter and have a capital letter for each new word.
Underscores (`_`) are not allowed.

The `using` syntax is preferred to `typedef` for type definitions.

### Namespace Names

If possible, namespace names should be one word, lowercase.
If multiple words are used they should be lowercase and separated with an underscores (`_`).

### Variable Names

#### Variable Names

Variable names should contain only lowercase characters and underscores (`_`).
The underscore is not allowed as first or last character.

#### Class Data Members

Class data members names follows the convention of variable names with a trailing underscore (`_`).

### Function Names

Function names should start with a lowercase character and have a capital letter for each new word.
Underscores (`_`) are not allowed.

The only exception to this rule are getter and setter methods for class data members.
They follow the naming convention `set_variable` and `get_variable`, respectively (see [Class Format](#class-format)).

### Lambda Expressions

Named lambda expressions follow the naming convention for variables:

```c++
auto my_lambda = [](int i) { return i + 4; };
```

### Macro Names

Macro names should be all uppercase and can include underscores (`_`).
The underscore is not allowed as first or last character.

### Naming Example

```c++
#define INCREMENT_MACRO(i) ++(i)

namespace two_words {
namespace name {

int variable_name;

void functionNameMultipleWords(...) {
  auto my_lambda = [](int i) { return i + 4; };
  other_statements;
}

class ClassNameMultipleWords {
public:
  ClassNameMultipleWords : member_name(0) {}

  int get_member_name() {
    return member_name_;
  }

  void set_member_name(int new_value) {
    member_name_ = new_value;
  }

protected:
  void functionName(Type parameter1, Type2 parameter2);

private:
  int member_name_;
};
}
}
```

## Comments

### Comment Style

Prefer the `// Comment` syntax, and use the `/* ... */` syntax only when is required.

Classes and functions has to be documented with doxigen using the `///` syntax.

### Function Comments

Each function should have a brief description on what the function does.

The detailed documentation should contain:
* for each function parameters whose type is non-const reference or pointer to non-const memory,
the specification if they are input (In:), output (Out:) or input-output parameters (InOut:),
* a description of the return value,
* the type of exception that can be thrown and when it happens.

Example:
```c++
/// \brief Updates foo and computes bar using in_1 .. in_5.
///
/// In: in_3, in_5
/// In/Out: foo
/// Out: bar
/// Returns the number of whatever.
/// Throws an exception of type MyExceptionType if bar cannot be computed.
int computeFooBar(Type in_1, const Type& in_2, Type& in_3,
                  const Type* in_4, Type* in_5, Type& foo,
                  Type& bar);
```

### Variable Comments

No comments, since the name should be self-descriptive.

### Class Comments

Every class should have a short description of what it is and what is does. If needed a detailed description can be added.
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

## Formatting

Use the provided clang-format style to format `.hpp`, `.ipp` and `.cpp` files.

### Line Length

The length of each line of your code should, in principle, be at most **105** characters.
This limit can be exceeded by few characters in special cases.

### Scopes

Do not use scopes for formatting reason.

### Horizontal Spacing

No trailing whitespaces should be added to any line.
Use no space before a comma (`,`) and a semicolon (`;`) and add a space after them if they are not at the end of a line.

### Parenthesis

Parenthesis should have no internal padding.
In general they have one space external padding, unless the following cases are met
which require no external padding:
* between two opening or two closing parenthesis,
* between the function name and its arguments,
* between a closing parenthesis and a comma (`,`) or a semicolon (`;`).

### Binary Operators

The assignment operator should always have spaces around it.
Other operators may have spaces around them, but it is not mandatory.

Note: clang-format forces a space before and one after the binary operators.

### Unary Operators

Do not put any space between an unary operator and their argument.

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

### Vertical Spacing

Use empty lines when it helps to improve the readability of the code, but do not use too many.
Do not use empty lines after a brace which opens a scope,
or before a brace which closes a scope.
Each file should contain an empty line at the end of the file.
Some editors add an empty line automatically, some do not.

### Indentation

Indentation consists of 2 spaces.
Do not use tabs in the code.

### Variable Declarations and Definitions

Do not declare multiple variables in the same declaration, especially if they are not fundamental types:

```c++
// Disallowed.
int x, y;
Object a("my-matrix"), b(10);

// Preferred way.
int x;
int y;
Object a("my-matrix");
Object b(10);
```

### Function Declarations and Definitions

The return type should be on the same line as the function name.
Parameters should be on the same line, too, unless they do not fit on it.
Include the parameter names also in the declaration of a function, i.e.
```c++
Type function(Type1 par1, Type2 par2, Type3 par3);
```
In function declarations comment the unused parameter names `Type /* unused_parameter_name */`

Examples:
```c++
Type Class::function(Type1 par1, Type2 par2) {
  statement;
  ...
}

Type LongNameClass::longNameFunction(Type1 par1, Type2 par2
                                     Type3 par3) {
  statement;
  ...
}
```

In case of a long list of parameters prefer
```c++
Type LongNameClass::longNameFunction(
    Type1 long_name_par1, Type2 long_name_par2,Type3 par3) {
  statement;
  ...
}
```
to
```c++
Type LongNameClass::longNameFunction(Type1 long_name_par1,
                                     Type2 long_name_par2,
                                     Type3 par3) {
  statement;
  ...
}
```

### Function Calls

Write the call on a single line if the length of the line does not exceed the maximum limit.
If it does, wrap the arguments at the parenthesis or start the arguments on a new line using 4 spaces indent.
Use the method which uses the smaller amount of lines.

Examples:
```c++
function(par1, par2, par3);

function(par1, par2,
         par3);

function(
    par1, par2, par3);
```

### Conditionals

Examples:
```c++
if (condition)
  statement;
else
  statement;

if (condition) {
  statement;
}
else if (condition2) {
  statement;
}
else {
  statement;
}
```

### Switch statement

Switch statements should always have a default case.

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

Examples:
```c++
for (statement; condition; statement)
  statement;

for (statement; condition; statement) {
  statement1;
  statement2;
}

while (condition)
  statement;

while (condition) {
  statement1;
  statement2;
}

do {
  statement;
}
while (condition);
```

### Preprocessor Directives

The preprocessor directives are not indented.
The hash is the first character of the line.

### Class Format

`public`, `protected` and `private` keywords are not indented.

Unless a different order is required, the members should be defined in the following order:
* using statements
* public members
* protected members
* private members

Example:
```c++
class Foo : public Bar {
public:
  Foo();
  explicit Foo(int var);

  void function();
  void emptyFunction() {}

  void set_var(const int var) {
    var_ = var;
  }
  int get_var() const {
    return var_;
  }

private:
  bool privateFunction();

  int var_;
  int var2_;
};
```

### Constructor Initializer Lists

Examples:
```c++
// When everything fits on one line:
Foo::Foo(int var) : var_(var) {
  statement;
}

// If the signature and the initializer list do not
// fit on one line, the colon is indented by 4 spaces:
Foo::Foo(int var)
    : var_(var), var2_(var + 1) {
  statement;
}

// If the initializer list occupies more lines,
// they are aligned in the following way:
Foo::Foo(int var)
    : some_var_(var),
      some_other_var_(var + 1) {
  statement;
}

// No statements:
Foo::Foo(int var)
    : some_var_(var) {}
```

## Other C++ Features

### Pre-increment and pre-decrement

Use the pre-increment (pre-decrement) operator when a variable is incremented (decremented) and the value of the expression is not used.
In particular, use the pre-increment (pre-decrement) operator for loop counters.

The post-increment and post-decrement operators create an unnecessary copy, that the compiler cannot optimize away in the case of iterators or other classes with overloaded increment and decrement operators.

### Alternative Operator Representations

Use the primary token for operators (E.g. `&&`, `||`, and `!`), alternative representations are not allowed (`and`, `or`, and `not`).
