ARG DEPS_IMAGE
FROM $DEPS_IMAGE

LABEL com.jfrog.artifactory.retention.maxDays="7"
LABEL com.jfrog.artifactory.retention.maxCount="10"

# Directory where the project is built
ARG BUILD=/DLA-Future-build
# Directory where the miniapps are built as separate project
ARG BUILD_MINIAPP=/DLA-Future-miniapp-build
# Directory where the sources are copied to
ARG SOURCE=/DLA-Future
# Directory for some helper executables
ARG BIN=/DLA-Future-build/bin

# Build DLA-Future
COPY . ${SOURCE}

SHELL ["/bin/bash", "-c"]

# Inject the coverage option in the spack package
RUN gawk -i inplace '$0 ~ "return args" {print "        args.append(self.define(\"DLAF_WITH_COVERAGE\", True))"} {print $0}' ${SOURCE}/spack/packages/dla-future/package.py

ARG NUM_PROCS
# Note: we force spack to build in ${BUILD} creating a link to it
RUN spack repo rm --scope site dlaf && \
    spack repo add ${SOURCE}/spack && \
    spack -e ci develop --no-clone --path ${SOURCE} --build-directory ${BUILD} dla-future@master && \
    spack -e ci concretize -f && \
    spack -e ci --config "config:flags:keep_werror:all" install --jobs ${NUM_PROCS} --keep-stage --verbose

RUN mkdir -p ${BIN} && cp -L ${SOURCE}/ci/{mpi-ctest,check-threads} ${BIN}

ARG PIP_OPTS
# pip is needed only to install fastcov (it is removed with
#     its dependencies after fastcov installation)
RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends python3-pip && \
    pip install ${PIP_OPTS} fastcov && \
    apt-get autoremove -qq -y python3-pip && \
    apt-get clean

RUN cd /usr/local/bin && \
  curl -Ls https://codecov.io/bash > codecov.sh && \
  echo "f0e7a3ee76a787c37aa400cf44aee0c9b473b2fa79092edfb36d1faa853bbe23 codecov.sh" | sha256sum --check --quiet && \
  chmod +x codecov.sh

# Make it easy to call our binaries.
ENV PATH="${BIN}:/usr/local/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2"

# Used in our ctest wrapper to upload reports
ENV ENABLE_COVERAGE="YES"

# Automatically print stacktraces on segfault
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libSegFault.so

WORKDIR ${BUILD}
