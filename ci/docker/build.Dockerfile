ARG BASE_IMAGE=ubuntu:24.04

FROM $BASE_IMAGE

# set jfrog autoclean policy
LABEL com.jfrog.artifactory.retention.maxDays="21"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/opt/spack/bin:/opt/libtree" \
    SPACK_COLOR=always
SHELL ["/bin/bash", "-c"]

ARG EXTRA_APTGET
RUN apt-get -yqq update && \
    apt-get -yqq install --no-install-recommends \
    software-properties-common \
    build-essential gfortran \
    autoconf automake libssl-dev ninja-build pkg-config \
    ${EXTRA_APTGET} \
    gawk \
    python3 python3-setuptools \
    git tar wget curl ca-certificates gpg-agent jq tzdata \
    patchelf unzip file gnupg2 libncurses-dev && \
    rm -rf /var/lib/apt/lists/*

# Install libtree for packaging
RUN mkdir -p /opt/libtree && \
    curl -Lfso /opt/libtree/libtree https://github.com/haampie/libtree/releases/download/v2.0.0/libtree_x86_64 && \
    chmod +x /opt/libtree/libtree

# Install MKL and remove static libs (to keep image smaller)
ARG USE_MKL=ON
ARG MKL_VERSION=2024.0
ARG MKL_SPEC=2024.0.0
RUN if [ "$USE_MKL" = "ON" ]; then \
      wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB 2>/dev/null > /etc/apt/trusted.gpg.d/intel.asc  && \
      apt-add-repository 'deb https://apt.repos.intel.com/oneapi all main' && \
      apt-get install -y -qq --no-install-recommends intel-oneapi-mkl-devel-${MKL_VERSION} && \
      rm -rf /var/lib/apt/lists/* && \
      find "/opt/intel/oneapi" -name "*.a" -delete ; \
    fi

# This is the spack version we want to have
ARG SPACK_SHA
ENV SPACK_SHA=$SPACK_SHA

# Install the specific ref of Spack provided by the user and find compilers
RUN mkdir -p /opt/spack && \
    curl -Ls "https://api.github.com/repos/spack/spack/tarball/$SPACK_SHA" | tar --strip-components=1 -xz -C /opt/spack

# Find compilers + Add gfortran to clang specs + Define which compiler we want to use
ARG COMPILER
RUN spack compiler find && \
    gawk -i inplace '$0 ~ "compiler:" {flag=0} $0 ~ "spec:.*clang" {flag=1} flag == 1 && $1 ~ "^f[c7]" && $2 ~ "null" {gsub("null","/usr/bin/gfortran",$0)} {print $0}' /root/.spack/linux/compilers.yaml && \
    spack config add "packages:all:require:[\"%${COMPILER}\"]"

RUN spack external find \
    autoconf \
    automake \
    bzip2 \
    cuda \
    diffutils \
    findutils \
    git \
    ninja \
    m4 \
    ncurses \
    openssl \
    perl \
    pkg-config \
    python \
    xz && \
    if [ "$USE_MKL" = "ON" ]; then \
      echo -e "  intel-oneapi-mkl:\n    externals:\n    - spec: \"intel-oneapi-mkl@$MKL_SPEC mpi_family=mpich\"\n      prefix: /opt/intel/oneapi\n    buildable: False" >> ~/.spack/packages.yaml ; \
    fi

# Add our custom spack repo from here
ARG SPACK_DLAF_REPO
COPY $SPACK_DLAF_REPO /user_repo

RUN spack repo add --scope site /user_repo

# Set this to a spack.yaml file which contains a spec
# e.g. --build-arg SPACK_ENVIRONMENT=ci/spack/my-env.yaml
ARG SPACK_ENVIRONMENT
ARG COMMON_SPACK_ENVIRONMENT
# Build dependencies
# 1. Create a spack environment named `ci` from the input spack.yaml file
COPY $SPACK_ENVIRONMENT /spack_environment/spack.yaml
COPY $COMMON_SPACK_ENVIRONMENT /spack_environment/
RUN spack env create --without-view ci /spack_environment/spack.yaml
# 2. Set the C++ standard
ARG CXXSTD=17
RUN spack -e ci config add "packages:dla-future:variants:cxxstd=${CXXSTD}"
# 3. Install only the dependencies of this (top level is our package)
ARG NUM_PROCS
RUN spack -e ci install --jobs ${NUM_PROCS} --fail-fast --only=dependencies

# make ctest executable available.
RUN ln -s `spack -e ci location -i cmake`/bin/ctest /usr/bin/ctest
