# Modified base image to allow building cray-mpich

## Preparation steps

```
mkdir lib64
cp -a /usr/lib64/libxpmem.* lib64/

git clone https://github.com/eth-cscs/alps-cluster-config.git

cp alps-cluster-config/beverin/packages.yaml packages.yaml
```

## Edit cluster config files

Modify `packages.yaml`:
```
   patchelf:
     require: "@:0.17"
   libfabric:
-    buildable: false
-    externals:
-    - spec: libfabric@1.15.2.0
-      prefix: /opt/cray/libfabric/1.15.2.0/
+    require: "@1.15.2"
   slurm:
     buildable: false
     externals:
```
Note: The container engine (CE) will replace libfabric with the system one when running the container.
Make sure to use the same version.

## Build and push

```
export TAG="v0.1"
CSCS_REGISTRY="jfrog.svc.cscs.ch/docker-ci-ext/4700071344751697"
podman login jfrog.svc.cscs.ch

podman build -f build.Dockerfile -t $CSCS_REGISTRY/base-images/rocm_6.0.2-dev-ubuntu-22.04:$TAG

podman push $CSCS_REGISTRY/base-images/rocm_6.0.2-dev-ubuntu-22.04:$TAG
```
