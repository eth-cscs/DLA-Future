//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "ns3c/memory/host.h"
#include "ns3c/tile.h"

#include "gtest/gtest.h"

typedef double T;

int m = 1024;
int n = 1024;
int ld = 1024;

TEST(TileTest, Constructor) {
  ns3c::memory::Host<T> tt(m * n);

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      *tt(i + ld * j) = i + 0.0001 * j;

  ns3c::Tile<ns3c::memory::Host<T>> mytile(m, n, tt, ld);

  for (int i = 0; i < mytile.m(); ++i)
    for (int j = 0; j < mytile.n(); ++j)
      EXPECT_EQ(mytile(i, j), *tt(i + ld * j));
}

TEST(TileTest, MoveConstructor) {
  ns3c::memory::Host<T> tt(m * n);

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      *tt(i + ld * j) = i + 0.0001 * j;

  ns3c::Tile<ns3c::memory::Host<T>> mytile(m, n, tt, ld);

  ns3c::Tile<ns3c::memory::Host<T>> mynewtile(std::move(mytile));

  for (int i = 0; i < mytile.m(); ++i)
    for (int j = 0; j < mytile.n(); ++j)
      EXPECT_EQ(mynewtile(i, j), *tt(i + ld * j));

  EXPECT_EQ(mytile.get_mem_ptr(), nullptr);
}

TEST(TileTest, SubtileConstructor) {
  int msub = 514;
  int nsub = 324;
  int isub = 124;
  int jsub = 633;

  ns3c::memory::Host<T> tt(m * n);

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      *tt(i + ld * j) = i + 0.0001 * j;

  ns3c::Tile<ns3c::memory::Host<T>> mytile(m, n, tt, ld);
  ns3c::Tile<ns3c::memory::Host<T>> mysubtile(msub, nsub, mytile.get_mem(isub, jsub), ld);

  for (int i = 0; i < mysubtile.m(); ++i)
    for (int j = 0; j < mysubtile.n(); ++j)
      EXPECT_EQ(mysubtile(i, j), mytile(i + isub, j + jsub));

  for (int i = 0; i < mysubtile.m(); ++i)
    for (int j = 0; j < mysubtile.n(); ++j)
      EXPECT_EQ(mysubtile.get_ptr(i, j), mytile.get_ptr(i + isub, j + jsub));
}
