
#include "ns3c/memory/host.h"

#include "gtest/gtest.h"

typedef double T;
int size=4096;

TEST(HostTest, Constructor) {
    
    
    T* test = new T[size];
    for(int i=0; i<size; ++i) test[i] = i;
    
    ns3c::memory::Host<T> tt(size);
    for(int i=0; i<size; ++i) *tt(i) = test[i];
    
    for(int i=0; i<size; ++i)
        EXPECT_EQ(*tt(i), test[i]);
}


TEST(HostTest, ConstructorFromPointer) {
    T* test = new T[size];
    for(int i=0; i<size; ++i) test[i] = i;
    
    ns3c::memory::Host<T> tt(size);
    for(int i=0; i<size; ++i) *tt(i) = test[i];
    
    ns3c::memory::Host<T> tt2(tt());
    
    for(int i=0; i<size; ++i)
        EXPECT_EQ(tt(i), tt2(i));
}
