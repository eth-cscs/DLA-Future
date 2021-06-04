// TODO:
// License ???

#pragma once

#include <cusolverDn.h>

// clang-format off
extern "C" {
cusolverStatus_t CUSOLVERAPI cusolverDnSsygst_bufferSize(
    cusolverDnHandle_t handle,
    int itype,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *B,
    int ldb,
    int *Lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygst_bufferSize(
    cusolverDnHandle_t handle,
    int itype,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *B,
    int ldb,
    int *Lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnChegst_bufferSize(
    cusolverDnHandle_t handle,
    int itype,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *B,
    int ldb,
    int *Lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegst_bufferSize(
    cusolverDnHandle_t handle,
    int itype,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *B,
    int ldb,
    int *Lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSsygst(
    cusolverDnHandle_t handle,
    int itype,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *B,
    int ldb,
    float *Workspace,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygst(
    cusolverDnHandle_t handle,
    int itype,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *B,
    int ldb,
    double *Workspace,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnChegst(
    cusolverDnHandle_t handle,
    int itype,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *B,
    int ldb,
    cuComplex *Workspace,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegst(
    cusolverDnHandle_t handle,
    int itype,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *B,
    int ldb,
    cuDoubleComplex *Workspace,
    int *devInfo);
}
// clang-format on
