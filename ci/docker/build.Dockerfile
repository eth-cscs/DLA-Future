ARG BASE_IMAGE=ubuntu:24.04

FROM $BASE_IMAGE

# set jfrog autoclean policy
LABEL com.jfrog.artifactory.retention.maxDays="21"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="$PATH:/opt/spack/bin" \
    SPACK_COLOR=always

# Overwrite entrypoint as NVIDIA images set a script that clog the output.
ENTRYPOINT []
CMD [ "/bin/bash" ]
SHELL ["/bin/bash", "-c"]

ARG EXTRA_APTGET
# python is needed for spack and fastcov
# codecov upload needs curl + ca-certificates
# glibc-tools is needed for libSegFault on ubuntu > 22.04
# jq, strace are needed for check-threads
# tzdata is needed to print correct time
RUN apt-get -yqq update && \
    apt-get -yqq install --no-install-recommends \
    software-properties-common \
    build-essential gfortran \
    autoconf automake libssl-dev ninja-build pkg-config \
    gawk git tar \
    wget curl ca-certificates gpg-agent tzdata \
    python3 python3-setuptools \
    glibc-tools jq strace \
    patchelf unzip file gnupg2 libncurses-dev \
    ${EXTRA_APTGET} && \
    rm -rf /var/lib/apt/lists/*

# Install MKL and remove static libs (to keep image smaller)
# ARG USE_MKL=ON
# ARG MKL_VERSION=2024.0
# ARG MKL_SPEC=2024.0.0
# RUN if [ "$USE_MKL" = "ON" ]; then \
#       wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB 2>/dev/null > /etc/apt/trusted.gpg.d/intel.asc  && \
#       apt-add-repository 'deb https://apt.repos.intel.com/oneapi all main' && \
#       apt-get install -y -qq --no-install-recommends intel-oneapi-mkl-devel-${MKL_VERSION} && \
#       rm -rf /var/lib/apt/lists/* && \
#       find "/opt/intel/oneapi" -name "*.a" -delete ; \
#     fi

# This is the spack version we want to have
ARG SPACK_SHA
ENV SPACK_SHA=$SPACK_SHA

# Install the specific ref of Spack provided by the user and find compilers
RUN mkdir -p /opt/spack && \
    curl -Ls "https://api.github.com/repos/spack/spack/tarball/$SPACK_SHA" | tar --strip-components=1 -xz -C /opt/spack

# Find compilers + Define which compiler we want to use
ARG COMPILER
RUN spack external find gcc llvm && \
    spack config add "packages:cxx:require:'${COMPILER}'" && \
    spack config add "packages:c:require:'${COMPILER}'" && \
    spack config add "packages:fortran:require:gcc"

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
    xz
    # && \
    # if [ "$USE_MKL" = "ON" ]; then \
    #   echo -e "  intel-oneapi-mkl:\n    externals:\n    - spec: \"intel-oneapi-mkl@$MKL_SPEC mpi_family=mpich\"\n      prefix: /opt/intel/oneapi\n    buildable: False" >> ~/.spack/packages.yaml ; \
    # fi

# Add our custom spack repo from here
ARG SPACK_DLAF_REPO
COPY $SPACK_DLAF_REPO /user_repo

RUN spack repo add --scope site /user_repo

### Workaround until CE provides full MPI substitution.
# Add ~/site/repo if it exists in the base image
RUN if [ -d ~/site/repo ]; then \
      # spack repo add --scope site ~/site/repo; \
      git clone -b spack-compilers-as-nodes --single-branch https://github.com/eth-cscs/alps-cluster-config ~/custom-site; \
      spack repo add --scope site ~/custom-site/site/repo; \
    fi

# Add languages as dependencies to cray-mpich to make sure it correctly finds
# compilers. Does nothing for the upstream cray-mpich.
RUN sed -i '/depends_on("patchelf/a \    depends_on("c")\n    depends_on("cxx")\n    depends_on("fortran")' \
      $(spack location -p cray-mpich)/package.py

# Set this to a spack.yaml file which contains a spec
# e.g. --build-arg SPACK_ENVIRONMENT=ci/spack/my-env.yaml
ARG SPACK_ENVIRONMENT
ARG COMMON_SPACK_ENVIRONMENT
ARG ENV_VIEW=/view

# Build dependencies
# 1. Create a spack environment named `ci` from the input spack.yaml file
COPY $SPACK_ENVIRONMENT /spack_environment/spack.yaml
COPY $COMMON_SPACK_ENVIRONMENT /spack_environment/
RUN spack env create --with-view ${ENV_VIEW} ci /spack_environment/spack.yaml
# 2. Set the C++ standard
ARG CXXSTD=17
RUN spack -e ci config add "packages:dla-future:variants:cxxstd=${CXXSTD}"
# 3. Concretize environment
RUN spack -e ci concretize
RUN spack -e ci spec -lI --cover edges
# 4. Install only the dependencies of this (top level is our package)
ARG NUM_PROCS
RUN spack -e ci install --jobs ${NUM_PROCS} --fail-fast --only=dependencies

# make ctest executable available.
RUN ln -s `spack -e ci location -i cmake`/bin/ctest /usr/bin/ctest

RUN echo ${ENV_VIEW}/lib > /etc/ld.so.conf.d/dlaf.conf && ldconfig
