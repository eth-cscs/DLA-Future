//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <exception>
#include <vector>

#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"

namespace dlaf {
namespace matrix {

/// A mirror of a source matrix on the target device.
///
/// Creates a copy of the source matrix on the target device on construction,
/// if needed. The source matrix is unchanged and accessible while the mirror
/// is alive. The mirror is copied back on destruction, if the source matrix
/// contains non-const elements.
template <class T, Device Target, Device Source>
class MatrixMirror;

/// A mirror of a source matrix on the target device, where the source and
/// target devices are the same, and the element type is const.
///
/// This specialization makes no copies of the source matrix, and acts only as
/// a reference to the source matrix.
template <class T, Device SourceTarget>
class MatrixMirror<const T, SourceTarget, SourceTarget> {
protected:
  Matrix<const T, SourceTarget>& mat_source;

public:
  /// Create a matrix mirror of the source matrix @p mat_source.
  MatrixMirror(Matrix<const T, SourceTarget>& mat_source) : mat_source(mat_source) {}

  /// Return a reference to the mirror matrix on the target device.
  Matrix<const T, SourceTarget>& get() {
    return mat_source;
  }
};

/// A mirror of a source matrix on the target device, where the source and
/// target devices are the same, and the element type is non-const.
///
/// This specialization makes no copies of the source matrix, and acts only as
/// a reference to the source matrix. This specialization additionally allows
/// getting read-write tiles from the mirror matrix.
template <class T, Device SourceTarget>
class MatrixMirror<T, SourceTarget, SourceTarget>
    : public MatrixMirror<const T, SourceTarget, SourceTarget> {
  using base_type = MatrixMirror<const T, SourceTarget, SourceTarget>;
  Matrix<T, SourceTarget>& mat_source;

public:
  /// Create a mirror of the source matrix @p mat_source.
  MatrixMirror(Matrix<T, SourceTarget>& mat_source) : base_type(mat_source), mat_source(mat_source) {}

  /// Return a reference to the mirror matrix on the target device.
  Matrix<T, SourceTarget>& get() {
    return mat_source;
  }

  /// Copies the source to the target matrix. Since the source and target
  /// devices are the same, this is a no-op.
  void syncSourceToTarget() {}

  /// Copies the target to the source matrix. Since the source and target
  /// devices are the same, this is a no-op.
  void syncTargetToSource() {}
};

/// A mirror of a source matrix on the target device, where the source and
/// target devices are the different, and the element type is const.
///
/// This specialization copies the source matrix to the target matrix on
/// construction. It does not copy the target matrix back to the source matrix on
/// destruction.
template <class T, Device Target, Device Source>
class MatrixMirror<const T, Target, Source> {
protected:
  Matrix<T, Target> mat_target;
  Matrix<const T, Source>& mat_source;

public:
  /// Create a mirror of the source matrix @p mat_source. Creates a copy of the
  /// source matrix on the target device.
  MatrixMirror(Matrix<const T, Source>& mat_source)
      : mat_target(mat_source.distribution()), mat_source(mat_source) {
    copy(mat_source, mat_target);
  }

  /// Release the target matrix.
  virtual ~MatrixMirror() = default;

  /// Return a reference to the mirror matrix on the target device.
  Matrix<const T, Target>& get() {
    return mat_target;
  }
};

/// A mirror of a source matrix on the target device, where the source and
/// target devices are the different, and the element type is non-const.
///
/// This specialization copies the source matrix to the target matrix on
/// construction, and copies the target matrix back to the source matrix on
/// destruction. The source and target matrices can also be explicitly
/// synchronized. This specialization additionally allows getting read-write
/// tiles from the target matrix.
template <class T, Device Target, Device Source>
class MatrixMirror : public MatrixMirror<const T, Target, Source> {
  using base_type = MatrixMirror<const T, Target, Source>;
  using base_type::mat_target;
  Matrix<T, Source>& mat_source;

public:
  /// Create a mirror of the source matrix @p mat_source. Creates a copy of the
  /// source matrix on the target device.
  MatrixMirror(Matrix<T, Source>& mat_source) : base_type(mat_source), mat_source(mat_source) {}

  /// Copy the target matrix back to the source matrix and release the target
  /// matrix.
  ~MatrixMirror() {
    syncTargetToSource();
  }

  /// Return a reference to the mirror matrix on the target device.
  Matrix<T, Target>& get() {
    return mat_target;
  }

  /// Copies the source to the target matrix.
  void syncSourceToTarget() {
    copy(mat_source, mat_target);
  }

  /// Copies the target to the source matrix.
  void syncTargetToSource() {
    copy(mat_target, mat_source);
  }
};

/// ---- ETI

#define DLAF_MATRIX_MIRROR_ETI(KWORD, DATATYPE, TARGETDEVICE, SOURCEDEVICE) \
  KWORD template class MatrixMirror<DATATYPE, TARGETDEVICE, SOURCEDEVICE>;  \
  KWORD template class MatrixMirror<const DATATYPE, TARGETDEVICE, SOURCEDEVICE>;

DLAF_MATRIX_MIRROR_ETI(extern, float, Device::CPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(extern, double, Device::CPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(extern, std::complex<float>, Device::CPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(extern, std::complex<double>, Device::CPU, Device::CPU)

#ifdef DLAF_WITH_CUDA
DLAF_MATRIX_MIRROR_ETI(extern, float, Device::CPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(extern, double, Device::CPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(extern, std::complex<float>, Device::CPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(extern, std::complex<double>, Device::CPU, Device::GPU)

DLAF_MATRIX_MIRROR_ETI(extern, float, Device::GPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(extern, double, Device::GPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(extern, std::complex<float>, Device::GPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(extern, std::complex<double>, Device::GPU, Device::CPU)

DLAF_MATRIX_MIRROR_ETI(extern, float, Device::GPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(extern, double, Device::GPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(extern, std::complex<float>, Device::GPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(extern, std::complex<double>, Device::GPU, Device::GPU)
#endif

}
}
