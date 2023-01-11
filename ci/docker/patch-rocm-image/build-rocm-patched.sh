#!/usr/bin/env bash

CSCS_REGISTRY="jfrog.svc.cscs.ch/contbuild/testing/anfink/4700071344751697"
docker build -t $CSCS_REGISTRY/rocm-patched:5.3.3 -f build.Dockerfile .
docker push $CSCS_REGISTRY/rocm-patched:5.3.3
