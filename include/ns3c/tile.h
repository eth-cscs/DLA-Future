//
// NS3C
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector>

namespace ns3c {

template <class Mem>
class Tile {
  using T = typename Mem::ElementType;

  public:
  Tile(int m, int n, Mem& mem, int ld) : m_(m), n_(n), mem_(&mem), ld_(ld) {}

  Tile(const Tile&) = delete;

  Tile(Tile&& rhs) : m_(rhs.m_), n_(rhs.n_), mem_(rhs.mem_), ld_(rhs.ld_) {
    rhs.mem_ = nullptr;
  }

  Tile& operator=(Tile&& rhs) {
    if (this != &rhs) {
      delete this->mem_;
      this->mem_ = rhs.mem_;
      rhs.mem_ = nullptr;

      this->m_ = rhs.m_;
      this->n_ = rhs.n_;
      this->ld_ = rhs.ld_;
    }
    return *this;
  }

  T& operator()() {
    return *((*mem_)());
  }

  const T& operator()() const {
    return *((*mem_)());
  }

  T& operator()(int i, int j) {
    return *((*mem_)(i + ld_ * j));
  }

  const T& operator()(int i, int j) const {
    return *((*mem_)(i + ld_ * j));
  }

  T* get_ptr() {
    return (*mem_)();
  }

  T* get_ptr(int i, int j) {
    return (*mem_)(i + ld_ * j);
  }

  const T* get_ptr(int i, int j) const {
    return (*mem_)(i + ld_ * j);
  }

  Mem& get_mem(int i, int j) {
    return *(new Mem(get_ptr(i, j)));
  }

  int m() const {
    return m_;
  }
  int n() const {
    return n_;
  }
  int ld() const {
    return ld_;
  }
  int size() const {
    return m * n;
  }

  private:
  int m_;
  int n_;
  int ld_;
  Mem* mem_;
};

}
