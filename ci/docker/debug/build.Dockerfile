FROM ubuntu:18.04

WORKDIR /root

ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_UNSAFE_CONFIGURE 1

# Install basic tools
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    software-properties-common \
    build-essential gfortran binutils \
    git tar wget curl gpg-agent jq tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install cmake
RUN wget -qO- "https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Install MPICH ABI compatible with Cray's lib on Piz Daint
ARG MPICH_VERSION=3.3.2
ARG MPICH_PATH=/usr/local/mpich
RUN wget -q https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar -xzf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure \
      --disable-fortran \
      --prefix=$MPICH_PATH && \
    make install -j$(nproc) && \
    rm -rf /root/mpich-${MPICH_VERSION}.tar.gz /root/mpich-${MPICH_VERSION}

# Install OpenBLAS
ARG OPENBLAS_MAJOR=0
ARG OPENBLAS_MINOR=3
ARG OPENBLAS_PATCH=9
RUN wget -qO - https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_MAJOR}.${OPENBLAS_MINOR}.${OPENBLAS_PATCH}.tar.gz -O openblas.tar.gz && \
    tar -xzf openblas.tar.gz && \
    cd OpenBLAS-${OPENBLAS_MAJOR}.${OPENBLAS_MINOR}.${OPENBLAS_PATCH}/ && \
    make USE_OPENMP=0 USE_THREAD=0 USE_LOCKING=1 DEBUG=1 -j$(nproc) && \
    make install NO_STATIC=1 PREFIX=/usr/local/ && \
    rm -rf /root/openblas.tar.gz /root/OpenBLAS-${OPENBLAS_MAJOR}.${OPENBLAS_MINOR}.${OPENBLAS_PATCH}/

# Install Boost
ARG BOOST_MAJOR=1
ARG BOOST_MINOR=72
ARG BOOST_PATCH=0
ARG BOOST_PATH=/usr/local/boost

RUN wget -q https://dl.bintray.com/boostorg/release/${BOOST_MAJOR}.${BOOST_MINOR}.${BOOST_PATCH}/source/boost_${BOOST_MAJOR}_${BOOST_MINOR}_${BOOST_PATCH}.tar.gz -O boost.tar.gz && \
    tar -xzf boost.tar.gz && \
    cd boost_${BOOST_MAJOR}_${BOOST_MINOR}_${BOOST_PATCH} && \
    ./bootstrap.sh --prefix=$BOOST_PATH && \
    ./b2 toolset=gcc variant=debug -j$(nproc) install && \
    rm -rf /root/boost.tar.gz /root/boost_${BOOST_MAJOR}_${BOOST_MINOR}_${BOOST_PATCH}

# Install hwloc
ARG HWLOC_MAJOR=2
ARG HWLOC_MINOR=2
ARG HWLOC_PATCH=0
ARG HWLOC_PATH=/usr/local/hwloc

RUN wget -q https://download.open-mpi.org/release/hwloc/v${HWLOC_MAJOR}.${HWLOC_MINOR}/hwloc-${HWLOC_MAJOR}.${HWLOC_MINOR}.${HWLOC_PATCH}.tar.gz -O hwloc.tar.gz && \
    tar -xzf hwloc.tar.gz && \
    cd hwloc-${HWLOC_MAJOR}.${HWLOC_MINOR}.${HWLOC_PATCH} && \
    ./configure --prefix=$HWLOC_PATH && \
    make -j$(nproc) install && \
    rm -rf /root/hwloc.tar.gz /root/hwloc-${HWLOC_MAJOR}.${HWLOC_MINOR}.${HWLOC_PATCH}

# Install tcmalloc (their version tagging is a bit inconsistent; patch version is not always included)
ARG GPERFTOOLS_VERSION=2.7
ARG GPERFTOOLS_PATH=/usr/local/gperftools
RUN wget -q https://github.com/gperftools/gperftools/releases/download/gperftools-${GPERFTOOLS_VERSION}/gperftools-${GPERFTOOLS_VERSION}.tar.gz -O gperftools.tar.gz && \
    tar -xzf gperftools.tar.gz && \
    cd gperftools-${GPERFTOOLS_VERSION} && \
    ./configure \
      --prefix=${GPERFTOOLS_PATH} && \
    make -j$(nproc) && \
    make install && \
    rm -rf /root/gperftools.tar.gz /root/gperftools-${GPERFTOOLS_VERSION}

# Install HPX
ARG HPX_FORK=STEllAR-GROUP
ARG HPX_VERSION=1.5.0
ARG HPX_PATH=/usr/local/hpx
RUN wget -q https://github.com/${HPX_FORK}/hpx/archive/${HPX_VERSION}.tar.gz -O hpx.tar.gz && \
    tar -xzf hpx.tar.gz && \
    cd hpx-${HPX_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=$HPX_PATH \
      -DBOOST_ROOT=$BOOST_PATH \
      -DHWLOC_ROOT=$HWLOC_PATH \
      -DTCMALLOC_ROOT=$GPERFTOOLS_PATH \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS_DEBUG="-g -Og -fno-omit-frame-pointer" \
      -DHPX_WITH_SANITIZERS=ON \
      -DHPX_WITH_STACK_OVERFLOW_DETECTION=OFF \
      -DHPX_WITH_MAX_CPU_COUNT=128 \
      -DHPX_WITH_NETWORKING=OFF \
      -DHPX_WITH_TESTS=OFF \
      -DHPX_WITH_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install && \
    rm -rf /root/hpx.tar.gz /root/hpx-${HPX_VERSION}

RUN ldconfig

# Install BLASPP
ARG BLASPP_VERSION=c090b5738c8e
ARG BLASPP_PATH=/usr/local/blaspp
RUN wget -q https://bitbucket.org/icl/blaspp/get/${BLASPP_VERSION}.tar.gz -O blaspp.tar.gz && \
    tar -xzf blaspp.tar.gz && \
    cd icl-blaspp-${BLASPP_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. \
      -DBLASPP_BUILD_TESTS=OFF \
      -DBLAS_LIBRARY=OpenBLAS \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS_DEBUG="-g -Og -fno-omit-frame-pointer" \
      -DCMAKE_INSTALL_PREFIX=$BLASPP_PATH \
      && \
    make -j$(nproc) && \
    make install && \
    rm -rf /root/blaspp.tar.gz /root/icl-blaspp-${BLASPP_VERSION}

ARG LAPACKPP_VERSION=f878fada3765
ARG LAPACKPP_PATH=/usr/local/lapackpp
RUN wget -q https://bitbucket.org/icl/lapackpp/get/${LAPACKPP_VERSION}.tar.gz -O lapackpp.tar.gz && \
   tar -xzf lapackpp.tar.gz && \
   cd icl-lapackpp-${LAPACKPP_VERSION} && \
   mkdir build && \
   cd build && \
   cmake .. \
     -DBUILD_LAPACKPP_TESTS=OFF \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_CXX_FLAGS_DEBUG="-g -Og -fno-omit-frame-pointer" \
     -DCMAKE_INSTALL_PREFIX=$LAPACKPP_PATH && \
   make -j$(nproc) install && \
   rm -rf /root/lapackpp.tar.gz /root/icl-lapackpp-${LAPACKPP_VERSION}

# Add deployment tooling
RUN wget -q https://github.com/haampie/libtree/releases/download/v1.1.3/libtree_x86_64.tar.gz && \
   tar -xzf libtree_x86_64.tar.gz && \
   rm libtree_x86_64.tar.gz && \
   ln -s /root/libtree/libtree /usr/local/bin/libtree
