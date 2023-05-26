//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/hdf5.h"

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "dlaf/common/index2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/error.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using dlaf::matrix::FileHDF5;

template <typename T>
class MatrixHDF5Test : public ::testing::Test {
protected:
  MatrixHDF5Test() : world(MPI_COMM_WORLD), filepath("test_matrix_hdf5.h5") {
    if (exists(filepath) && isMasterRank())
      std::filesystem::remove(filepath);
  }

  ~MatrixHDF5Test() override {
    if (exists(filepath) && isMasterRank())
      std::filesystem::remove(filepath);
  }

  bool isMasterRank() const {
    return this->world.rank() == (world.size() / 2);
  }

  comm::Communicator world;
  const std::filesystem::path filepath;
  const std::string dataset_name = "/matrix";
};

TYPED_TEST_SUITE(MatrixHDF5Test, dlaf::test::MatrixElementTypes);

template <class T, Device D, class Func>
void testReadLocal(const FileHDF5& file, const std::string& dataset_name,
                   const Matrix<const T, D>& mat_original, Func&& original_values) {
  Matrix<const T, D> mat_local = file.read<T, D>(dataset_name, {13, 27});

  EXPECT_TRUE(local_matrix(mat_local));
  EXPECT_EQ(mat_original.size(), mat_local.size());
  EXPECT_EQ(TileElementSize(13, 27), mat_local.blockSize());

  dlaf::matrix::MatrixMirror<const T, Device::CPU, D> matrix_host(mat_local);
  CHECK_MATRIX_EQ(original_values, matrix_host.get());
}

template <class T, Device D, class Func>
void testReadDistributed(const FileHDF5& file, const std::string& dataset_name,
                         comm::CommunicatorGrid grid, const Matrix<const T, D>& mat_original,
                         Func&& original_values) {
  Matrix<const T, D> mat_dist = file.read<T, D>(dataset_name, {26, 13}, grid, {0, 0});

  EXPECT_EQ(grid.size(), mat_dist.distribution().commGridSize());
  EXPECT_EQ(grid.rank(), mat_dist.distribution().rankIndex());
  EXPECT_EQ(mat_original.size(), mat_dist.size());
  EXPECT_EQ(TileElementSize(26, 13), mat_dist.blockSize());

  dlaf::matrix::MatrixMirror<const T, Device::CPU, D> matrix_host(mat_dist);
  CHECK_MATRIX_EQ(original_values, matrix_host.get());
}

template <class T, Device D>
auto getLocalMatrixAndSetter() {
  auto original_values = [](const GlobalElementIndex& ij) { return ij.row() * 100 + ij.col(); };

  matrix::Matrix<T, Device::CPU> matrix({83, 88}, {9, 6});
  dlaf::matrix::util::set(matrix, original_values);

  if constexpr (D == Device::CPU)
    return std::make_tuple(std::move(matrix), std::move(original_values));

  Matrix<T, D> mat_device(matrix.distribution());
  copy(matrix, mat_device);
  return std::make_tuple(std::move(mat_device), std::move(original_values));
}

template <class T, Device D>
void testHDF5(bool isMasterRank, comm::Communicator world, const std::filesystem::path& filepath,
              const std::string& dataset_name) {
  comm::CommunicatorGrid grid(world, world.size(), 1, common::Ordering::RowMajor);

  auto [mat_original, original_values] = getLocalMatrixAndSetter<T, D>();

  // Just one of the two ranks write the file, the other waits the MPI barrier which ensures that
  // file has been written on disk.
  if (isMasterRank) {
    FileHDF5 file(filepath, FileHDF5::FileMode::READWRITE);
    file.write(mat_original, dataset_name);

    // Verify that with a READWRITE file it is possible to read
    testReadLocal(file, dataset_name, mat_original, original_values);
  }
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  // At this point all ranks can open the file independently in readonly mode.
  FileHDF5 file(filepath, FileHDF5::FileMode::READONLY);

  testReadLocal(file, dataset_name, mat_original, original_values);
  testReadDistributed(file, dataset_name, grid, mat_original, original_values);
}

TYPED_TEST(MatrixHDF5Test, SingleWriteMC) {
  testHDF5<TypeParam, Device::CPU>(this->isMasterRank(), this->world, this->filepath,
                                   this->dataset_name);
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(MatrixHDF5Test, SingleWriteGPU) {
  testHDF5<TypeParam, Device::GPU>(this->isMasterRank(), this->world, this->filepath,
                                   this->dataset_name);
}
#endif

template <class T, Device D>
void testHDF5Parallel(const bool isMasterRank, comm::Communicator world,
                      const std::filesystem::path& filepath, const std::string& dataset_name) {
  comm::CommunicatorGrid grid(world, world.size(), 1, common::Ordering::RowMajor);

  auto&& [mat_original, original_values] = getLocalMatrixAndSetter<T, D>();

  FileHDF5 file(world, filepath);
  file.write(mat_original, dataset_name);

  // Note:
  // Before reading, ensures that all ranks have flushed buffers and synchronize with a barrier.
  file.flush();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  testReadLocal(file, dataset_name, mat_original, original_values);
  testReadDistributed(file, dataset_name, grid, mat_original, original_values);

  // Check that a single rank can read, without involving others
  if (isMasterRank)
    testReadLocal(file, dataset_name, mat_original, original_values);
}

TYPED_TEST(MatrixHDF5Test, RWParallelMC) {
  testHDF5Parallel<TypeParam, Device::CPU>(this->isMasterRank(), this->world, this->filepath,
                                           this->dataset_name);
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(MatrixHDF5Test, RWParallelGPU) {
  testHDF5Parallel<TypeParam, Device::GPU>(this->isMasterRank(), this->world, this->filepath,
                                           this->dataset_name);
}
#endif
