//
// NS3C
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "gtest/gtest.h"
#include "ns3c/memory/host.h"

typedef double T;
int size = 4096;

TEST(HostTest, Constructor) {
  T* test = new T[size];
  for (int i = 0; i < size; ++i)
    test[i] = i;

  ns3c::memory::Host<T> tt(size);
  for (int i = 0; i < size; ++i)
    *tt(i) = test[i];

  for (int i = 0; i < size; ++i)
    EXPECT_EQ(*tt(i), test[i]);
}

TEST(HostTest, ConstructorFromPointer) {
  T* test = new T[size];
  for (int i = 0; i < size; ++i)
    test[i] = i;

  ns3c::memory::Host<T> tt(size);
  for (int i = 0; i < size; ++i)
    *tt(i) = test[i];

  ns3c::memory::Host<T> tt2(tt());

  for (int i = 0; i < size; ++i)
    EXPECT_EQ(tt(i), tt2(i));
}

TEST(HostTest, MoveConstructor) {
  T* test = new T[size];
  for (int i = 0; i < size; ++i)
    test[i] = i;
  ns3c::memory::Host<T> tt(size);
  for (int i = 0; i < size; ++i)
    *tt(i) = test[i];

  ns3c::memory::Host<T> tt2(std::move(tt));

  for (int i = 0; i < size; ++i)
    EXPECT_EQ(*tt2(i), test[i]);

  EXPECT_EQ(tt(), nullptr);
}

TEST(HostTest, MoveAssignement) {
  T* test = new T[size];
  for (int i = 0; i < size; ++i)
    test[i] = i;
  ns3c::memory::Host<T> tt(size);
  for (int i = 0; i < size; ++i)
    *tt(i) = test[i];

  ns3c::memory::Host<T> tt2 = std::move(tt);

  for (int i = 0; i < size; ++i)
    EXPECT_EQ(*tt2(i), test[i]);

  EXPECT_EQ(tt(), nullptr);
}
