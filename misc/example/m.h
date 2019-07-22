#pragma once

#include "out.h"

#include <array>
#include <hpx/hpx.hpp>
#include <iostream>
#include <memory>

template <class El>
class Tile {
  public:
  Tile(El* ptr) : ptr_(ptr) {}

  Tile(const Tile&) = delete;
  Tile(Tile&& rhs) : ptr_(rhs.ptr_) {
    //std::cout << "Move" << std::endl;
    rhs.ptr_ = nullptr;
  }

  El& operator()() {
    return *ptr_;
  }

  const El& operator()() const {
    return *ptr_;
  }

  private:
  El* ptr_;
};

template <class Type>
class Wrapper {
  public:
  Wrapper(Type&& t, hpx::promise<Type>&& p) : t_(std::move(t)), p_(std::move(p)), valid_(true) {}

  ~Wrapper() {
    if (valid_) {
      p_.set_value(std::move(t_));
    }
  }

  Wrapper(const Wrapper&) = delete;
  Wrapper(Wrapper&& rhs) : t_(std::move(rhs.t_)), p_(std::move(rhs.p_)), valid_(true) {
    rhs.valid_ = false;
  }

  Type& get() {
    return t_;
  }
  const Type& get() const {
    return t_;
  }

  private:
  Type t_;
  hpx::promise<Type> p_;
  bool valid_;
};

template <class Type>
class MatrixRW;
template <class Type>
class MatrixRead;

template <class Type>
class ConstMatrix {
  public:
  ConstMatrix(std::array<hpx::future<Type>, 4>&& f) : f_(std::move(f)) {}

  ConstMatrix(ConstMatrix&& rhs) : f_(std::move(rhs.f_)), s_(std::move(rhs.s_)){};

  hpx::shared_future<Wrapper<Type>> read(int i) const {
    // if the shared future is not valid (i.e. the previous task modified the tile)
    // a new shared future is created. when all the shared future are destroyed, the Wrapper object
    // is destroyed as well and the future is set allowing write operation to go on.
    // Note: the copy of the i-th shared future hold by *this is destroyed when the operator(i) is called.
    if (!s_[i].valid()) {
      hpx::future<Type> fut = std::move(f_[i]);
      hpx::promise<Type> p;
      f_[i] = p.get_future();
      s_[i] = std::move(fut.then(hpx::launch::sync, [p = std::move(p)](hpx::future<Type>&& fut) mutable {
        return Wrapper(std::move(fut.get()), std::move(p));
      }));
    }
    return s_[i];
  }

  MatrixRead<Type> block_read();

  protected:
  // used for building RW matrix.
  ConstMatrix(std::array<hpx::future<Type>, 4>&& f, std::array<hpx::shared_future<Wrapper<Type>>, 4>&& s)
      : f_(std::move(f)), s_(std::move(s)) {}
  // used for building read-only matrix.
  ConstMatrix(std::array<hpx::shared_future<Wrapper<Type>>, 4>&& s) : f_(), s_(std::move(s)) {
    for (std::size_t i = 0; i < s_.size(); ++i) {
      if (!s_[i].valid()) {
        std::cerr << "ERROR: Invalid shared future!" << std::endl;
        hpx::terminate();
      }
    }
  }

  mutable std::array<hpx::future<Type>, 4> f_;
  mutable std::array<hpx::shared_future<Wrapper<Type>>, 4> s_;
};

template <class Type>
class Matrix : public ConstMatrix<Type> {
  protected:
  using ConstMatrix<Type>::f_;
  using ConstMatrix<Type>::s_;

  public:
  Matrix(std::array<hpx::future<Type>, 4>&& f) : ConstMatrix<Type>(std::move(f)) {}

  Matrix(Matrix&& rhs) : ConstMatrix<Type>(std::move(rhs)){};

  // Create a new future for i-th tile which will be set as ready when the Wrapper object included in the returned future is destroyed.
  hpx::future<Wrapper<Type>> operator()(int i) {
    auto fut = std::move(f_[i]);
    hpx::promise<Type> p;
    f_[i] = p.get_future();
    s_[i] = {};
    return fut.then(hpx::launch::sync, [p = std::move(p)](hpx::future<Type>&& fut) mutable {
      return Wrapper(std::move(fut.get()), std::move(p));
    });
  }

  MatrixRW<Type> block();

  protected:
  // used for building RW matrix.
  Matrix(std::array<hpx::future<Type>, 4>&& f, std::array<hpx::shared_future<Wrapper<Type>>, 4>&& s)
      : ConstMatrix<Type>(std::move(f), std::move(s)) {}
};

template <class Type>
class MatrixRW : public Matrix<Type> {
  protected:
  using Matrix<Type>::f_;
  using Matrix<Type>::s_;

  public:
  MatrixRW(std::array<hpx::future<Type>, 4>&& f, std::array<hpx::shared_future<Wrapper<Type>>, 4>&& s,
           std::array<std::unique_ptr<hpx::promise<Type>>, 4>&& p,
           std::array<std::unique_ptr<hpx::promise<Wrapper<Type>>>, 4>&& sp,
           std::array<hpx::shared_future<Wrapper<Type>>, 4> sf)
      : Matrix<Type>(std::move(f), std::move(s)), p_(std::move(p)), sp_(std::move(sp)), sf_(sf) {}

  ~MatrixRW() {
    for (std::size_t i = 0; i < p_.size(); ++i) {
      done(i);
    }
  }

  MatrixRW(MatrixRW&&) = default;

  void doneWrite(int i) {
    if (p_[i]) {
      s_[i] = sf_[i];
      f_[i].then(hpx::launch::sync,
                 [p = std::move(p_[i]), sp = std::move(sp_[i])](hpx::future<Type>&& fut) mutable {
                   sp->set_value(Wrapper(std::move(fut.get()), std::move(*p)));
                 });
      p_[i] = nullptr;
      sp_[i] = nullptr;
    }
  }

  void done(int i) {
    doneWrite(i);
    s_[i] = {};
  }

  private:
  std::array<std::unique_ptr<hpx::promise<Type>>, 4> p_;
  std::array<std::unique_ptr<hpx::promise<Wrapper<Type>>>, 4> sp_;
  std::array<hpx::shared_future<Wrapper<Type>>, 4> sf_;
};

template <class Type>
class MatrixRead : public ConstMatrix<Type> {
  protected:
  using ConstMatrix<Type>::s_;

  public:
  MatrixRead(std::array<hpx::shared_future<Wrapper<Type>>, 4>&& s) : ConstMatrix<Type>(std::move(s)) {}

  MatrixRead(MatrixRead&&) = default;

  ~MatrixRead() {
    for (std::size_t i = 0; i < s_.size(); ++i) {
      done(i);
    }
  }

  void done(int i) {
    if (s_[i].valid()) {
      //std::cout << "done " << i << std::endl;
      s_[i] = {};
    }
  }
};

template <class Type>
MatrixRW<Type> Matrix<Type>::block() {
  // Create a new future for each tile. The i-th future will be set as ready when the done(i)
  // method of MatrixRW is called or the MatrixRW object is destroyed.
  // The current futures and shared futures are moved to the MatrixRW object.
  std::array<std::unique_ptr<hpx::promise<Type>>, 4> p;
  std::array<std::unique_ptr<hpx::promise<Wrapper<Type>>>, 4> sp;

  std::array<hpx::future<Type>, 4> f = std::move(f_);
  std::array<hpx::shared_future<Wrapper<Type>>, 4> s = std::move(s_);

  for (std::size_t i = 0; i < f_.size(); ++i) {
    p[i] = std::make_unique<hpx::promise<Type>>();
    sp[i] = std::make_unique<hpx::promise<Wrapper<Type>>>();
    f_[i] = std::move(p[i]->get_future());
    s_[i] = sp[i]->get_future();
  }
  return MatrixRW<Type>(std::move(f), std::move(s), std::move(p), std::move(sp), s_);
}

template <class Type>
MatrixRead<Type> ConstMatrix<Type>::block_read() {
  // Create if not already available the shared future for the tiles and store a copy of them
  // in the new MatrixRead object.
  std::array<hpx::shared_future<Wrapper<Type>>, 4> s;
  for (std::size_t i = 0; i < f_.size(); ++i) {
    s[i] = std::move(read(i));
  }
  return MatrixRead<Type>(std::move(s));
}
