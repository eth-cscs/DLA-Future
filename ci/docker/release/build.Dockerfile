ARG BASE_IMAGE=ubuntu:20.04

FROM $BASE_IMAGE

WORKDIR /root

SHELL ["/bin/bash", "-l", "-c"]

ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_UNSAFE_CONFIGURE 1

# Install basic tools
RUN apt-get update -qq && apt-get install -qq --no-install-recommends \
    software-properties-common \
    build-essential \
    git tar wget curl gpg-agent jq tzdata libasio-dev && \
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

# Install MKL
ARG MKL_VERSION=2020.0-088
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB 2>/dev/null | apt-key add - && \
    apt-add-repository 'deb https://apt.repos.intel.com/mkl all main' && \
    apt-get install -y -qq --no-install-recommends intel-mkl-64bit-${MKL_VERSION} && \
    rm -rf /var/lib/apt/lists/* && \
    echo "/opt/intel/lib/intel64\n/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && \
    ldconfig

# Install Boost
ARG BOOST_MAJOR=1
ARG BOOST_MINOR=72
ARG BOOST_PATCH=0
ARG BOOST_PATH=/usr/local/boost

RUN wget -q https://boostorg.jfrog.io/artifactory/main/release/${BOOST_MAJOR}.${BOOST_MINOR}.${BOOST_PATCH}/source/boost_${BOOST_MAJOR}_${BOOST_MINOR}_${BOOST_PATCH}.tar.gz -O boost.tar.gz && \
    tar -xzf boost.tar.gz && \
    cd boost_${BOOST_MAJOR}_${BOOST_MINOR}_${BOOST_PATCH} && \
    ./bootstrap.sh --prefix=$BOOST_PATH && \
    ./b2 -j$(nproc) debug-symbols=on install && \
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
ARG HPX_VERSION=1.7.0
ARG HPX_WITH_CUDA=OFF
ARG HPX_PATH=/usr/local/hpx
RUN wget -q https://github.com/${HPX_FORK}/hpx/archive/${HPX_VERSION}.tar.gz -O hpx.tar.gz && \
    tar -xzf hpx.tar.gz && \
    cd hpx-${HPX_VERSION} && \
    mkdir build && \
    cd build && \
    CXX=${MPICH_PATH}/bin/mpicxx CC=${MPICH_PATH}/bin/mpicc cmake .. \
      -DBOOST_ROOT=$BOOST_PATH \
      -DHWLOC_ROOT=$HWLOC_PATH \
      -DTCMALLOC_ROOT=$GPERFTOOLS_PATH \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_INSTALL_PREFIX=$HPX_PATH \
      -DHPX_WITH_MAX_CPU_COUNT=128 \
      -DHPX_WITH_NETWORKING=OFF \
      -DHPX_WITH_ASYNC_MPI=ON \
      -DHPX_WITH_CUDA=$HPX_WITH_CUDA \
      -DHPX_WITH_TESTS=OFF \
      -DHPX_WITH_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install && \
    rm -rf /root/hpx.tar.gz /root/hpx-${HPX_VERSION}

ARG UMPIRE_VERSION=5.0.1
ARG UMPIRE_PATH=/usr/local/umpire
ARG UMPIRE_ENABLE_CUDA=ON
RUN git clone --recursive --depth 1 --branch v${UMPIRE_VERSION} https://github.com/LLNL/Umpire.git Umpire-${UMPIRE_VERSION} && \
    cd Umpire-${UMPIRE_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_INSTALL_PREFIX=$UMPIRE_PATH \
      -DENABLE_CUDA=$UMPIRE_ENABLE_CUDA \
      -DENABLE_BENCHMARKS=OFF \
      -DENABLE_TESTS=OFF && \
    make -j$(nproc) && \
    make install && \
    rm -rf /root/umpire.tar.gz /root/Umpire-${UMPIRE_VERSION}

# Install BLASPP
ARG BLASPP_VERSION=2020.10.02
ARG BLASPP_PATH=/usr/local/blaspp
RUN source /opt/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64 && \
    wget -q https://bitbucket.org/icl/blaspp/downloads/blaspp-${BLASPP_VERSION}.tar.gz -O blaspp.tar.gz && \
    tar -xzf blaspp.tar.gz && \
    cd blaspp-${BLASPP_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. \
      -Dbuild_tests=OFF \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -Duse_openmp=OFF \
      -Duse_cuda=OFF \
      -Dblas='Intel MKL' \
      -Dblas_threaded=OFF \
      -DCMAKE_INSTALL_PREFIX=$BLASPP_PATH && \
    make -j$(nproc) && \
    make install && \
    rm -rf /root/blaspp.tar.gz /root/blaspp-${BLASPP_VERSION}

ARG LAPACKPP_VERSION=2020.10.02
ARG LAPACKPP_PATH=/usr/local/lapackpp
RUN source /opt/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64 && \
    wget -q https://bitbucket.org/icl/lapackpp/downloads/lapackpp-$LAPACKPP_VERSION.tar.gz -O lapackpp.tar.gz && \
    tar -xzf lapackpp.tar.gz && \
    cd lapackpp-${LAPACKPP_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. \
      -Dbuild_tests=OFF \
      -DCMAKE_INSTALL_PREFIX=$LAPACKPP_PATH && \
    make -j$(nproc) install && \
    rm -rf /root/lapackpp.tar.gz /root/lapackpp-${LAPACKPP_VERSION}

# Add deployment tooling
RUN wget -q https://github.com/haampie/libtree/releases/download/v1.2.0/libtree_x86_64.tar.gz && \
    tar -xzf libtree_x86_64.tar.gz && \
    rm libtree_x86_64.tar.gz && \
    ln -s /root/libtree/libtree /usr/local/bin/libtree
