#pragma once

#include "out.h"

#include <array>
#include <cassert>
#include <hpx/hpx.hpp>
#include <iostream>
#include <memory>

template <class El>
class Tile {
  using ncEl = std::remove_const_t<El>;
  using cEl = std::add_const_t<El>;
  friend Tile<ncEl>;
  friend Tile<cEl>;

public:
  Tile(El* ptr) : ptr_(ptr), p_(nullptr) {}

  template <class T = El>
  Tile(std::enable_if_t<std::is_same<T, El>::value && !std::is_const<T>::value, El*> ptr,
       hpx::lcos::local::promise<Tile<El>>&& p)
      : ptr_(ptr), p_(std::make_unique<hpx::lcos::local::promise<Tile>>(std::move(p))) {}

  Tile(const Tile&) = delete;
  Tile(Tile&& rhs) = default;

  template <class T = El>
  Tile(std::enable_if_t<std::is_same<T, El>::value && std::is_const<T>::value, Tile<ncEl>&&> rhs)
      : ptr_(rhs.ptr_), p_(std::move(rhs.p_)) {}

  ~Tile() {
    if (p_) {
      auto p = std::move(p_);
      p->set_value(Tile<ncEl>(ptr_));
    }
  }

  template <class T = El>
  std::enable_if_t<std::is_same<T, El>::value && !std::is_const<T>::value, Tile&> setPromise(
      hpx::lcos::local::promise<Tile<El>>&& p) {
    assert(!p_);
    p_ = std::make_unique<hpx::lcos::local::promise<Tile<El>>>(std::move(p));
    return *this;
  }

  El& operator()() const {
    return *ptr_;
  }

private:
  ncEl* ptr_;
  std::unique_ptr<hpx::lcos::local::promise<Tile<ncEl>>> p_;
};

template <class Type>
class MatrixRW;
template <class Type>
class MatrixRead;

template <class El>
class ConstMatrix {
  using Type = Tile<El>;
  using ConstType = Tile<const El>;

public:
  ConstMatrix(std::array<hpx::future<Type>, 4>&& f) : f_(std::move(f)) {}

  ConstMatrix(ConstMatrix&& rhs) : f_(std::move(rhs.f_)), s_(std::move(rhs.s_)){};

  hpx::shared_future<ConstType> read(int i) const {
    // if the shared future is not valid (i.e. the previous task modified the tile)
    // a new shared future is created. when all the shared future are destroyed, the Wrapper object
    // is destroyed as well and the future is set allowing write operation to go on.
    // Note: the copy of the i-th shared future hold by *this is destroyed when the operator(i) is called.
    if (!s_[i].valid()) {
      hpx::future<Type> fut = std::move(f_[i]);
      hpx::lcos::local::promise<Type> p;
      f_[i] = p.get_future();
      s_[i] = std::move(fut.then(hpx::launch::sync, [p = std::move(p)](hpx::future<Type>&& fut) mutable {
        return ConstType(std::move(fut.get().setPromise(std::move(p))));
      }));
    }
    return s_[i];
  }

  MatrixRead<El> block_read();

protected:
  // used for building RW matrix.
  ConstMatrix(std::array<hpx::future<Type>, 4>&& f, std::array<hpx::shared_future<ConstType>, 4>&& s)
      : f_(std::move(f)), s_(std::move(s)) {}
  // used for building read-only matrix.
  ConstMatrix(std::array<hpx::shared_future<ConstType>, 4>&& s) : f_(), s_(std::move(s)) {
    for (std::size_t i = 0; i < s_.size(); ++i) {
      if (!s_[i].valid()) {
        std::cerr << "ERROR: Invalid shared future!" << std::endl;
        hpx::terminate();
      }
    }
  }

  mutable std::array<hpx::future<Type>, 4> f_;
  mutable std::array<hpx::shared_future<ConstType>, 4> s_;
};

template <class El>
class Matrix : public ConstMatrix<El> {
  using Type = Tile<El>;
  using ConstType = Tile<const El>;

protected:
  using ConstMatrix<El>::f_;
  using ConstMatrix<El>::s_;

public:
  Matrix(std::array<hpx::future<Type>, 4>&& f) : ConstMatrix<El>(std::move(f)) {}

  Matrix(Matrix&& rhs) : ConstMatrix<El>(std::move(rhs)){};

  // Create a new future for i-th tile which will be set as ready when the Wrapper object included in the
  // returned future is destroyed.
  hpx::future<Type> operator()(int i) {
    auto fut = std::move(f_[i]);
    hpx::lcos::local::promise<Type> p;
    f_[i] = p.get_future();
    s_[i] = {};
    return fut.then(hpx::launch::sync, [p = std::move(p)](hpx::future<Type>&& fut) mutable {
      return std::move(fut.get().setPromise(std::move(p)));
    });
  }

  MatrixRW<El> block();

protected:
  // used for building RW matrix.
  Matrix(std::array<hpx::future<Type>, 4>&& f, std::array<hpx::shared_future<ConstType>, 4>&& s)
      : ConstMatrix<El>(std::move(f), std::move(s)) {}
};

template <class El>
class MatrixRW : public Matrix<El> {
  using Type = Tile<El>;
  using ConstType = Tile<const El>;

protected:
  using Matrix<El>::f_;
  using Matrix<El>::s_;

public:
  MatrixRW(std::array<hpx::future<Type>, 4>&& f, std::array<hpx::shared_future<ConstType>, 4>&& s,
           std::array<std::unique_ptr<hpx::lcos::local::promise<Type>>, 4>&& p,
           std::array<std::unique_ptr<hpx::lcos::local::promise<ConstType>>, 4>&& sp,
           std::array<hpx::shared_future<ConstType>, 4> sf)
      : Matrix<El>(std::move(f), std::move(s)), p_(std::move(p)), sp_(std::move(sp)), sf_(sf) {}

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
                   sp->set_value(ConstType(std::move(fut.get().setPromise(std::move(*p)))));
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
  std::array<std::unique_ptr<hpx::lcos::local::promise<Type>>, 4> p_;
  std::array<std::unique_ptr<hpx::lcos::local::promise<ConstType>>, 4> sp_;
  std::array<hpx::shared_future<ConstType>, 4> sf_;
};

template <class El>
class MatrixRead : public ConstMatrix<El> {
  using ConstType = Tile<const El>;

protected:
  using ConstMatrix<El>::s_;

public:
  MatrixRead(std::array<hpx::shared_future<ConstType>, 4>&& s) : ConstMatrix<El>(std::move(s)) {}

  MatrixRead(MatrixRead&&) = default;

  ~MatrixRead() {
    for (std::size_t i = 0; i < s_.size(); ++i) {
      done(i);
    }
  }

  void done(int i) {
    if (s_[i].valid()) {
      // std::cout << "done " << i << std::endl;
      s_[i] = {};
    }
  }
};

template <class El>
MatrixRW<El> Matrix<El>::block() {
  // Create a new future for each tile. The i-th future will be set as ready when the done(i)
  // method of MatrixRW is called or the MatrixRW object is destroyed.
  // The current futures and shared futures are moved to the MatrixRW object.
  std::array<std::unique_ptr<hpx::lcos::local::promise<Type>>, 4> p;
  std::array<std::unique_ptr<hpx::lcos::local::promise<ConstType>>, 4> sp;

  std::array<hpx::future<Type>, 4> f = std::move(f_);
  std::array<hpx::shared_future<ConstType>, 4> s = std::move(s_);

  for (std::size_t i = 0; i < f_.size(); ++i) {
    p[i] = std::make_unique<hpx::lcos::local::promise<Type>>();
    sp[i] = std::make_unique<hpx::lcos::local::promise<ConstType>>();
    f_[i] = std::move(p[i]->get_future());
    s_[i] = sp[i]->get_future();
  }
  return MatrixRW<El>(std::move(f), std::move(s), std::move(p), std::move(sp), s_);
}

template <class El>
MatrixRead<El> ConstMatrix<El>::block_read() {
  // Create if not already available the shared future for the tiles and store a copy of them
  // in the new MatrixRead object.
  std::array<hpx::shared_future<ConstType>, 4> s;
  for (std::size_t i = 0; i < f_.size(); ++i) {
    s[i] = std::move(read(i));
  }
  return MatrixRead<El>(std::move(s));
}
