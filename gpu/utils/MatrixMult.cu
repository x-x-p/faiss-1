/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/DeviceMemory.h>

#include <cublas_v2.h>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>

namespace faiss { namespace gpu {

class DeviceMemory;

template <typename T>
struct GetCudaType;

template <>
struct GetCudaType<float> {
    static constexpr cudaDataType_t Type = CUDA_R_32F;
};

template <>
struct GetCudaType<half> {
    static constexpr cudaDataType_t Type = CUDA_R_16F;
};

template <>
struct GetCudaType<int8_t> {
    static constexpr cudaDataType_t Type = CUDA_R_8I;
};

template <>
struct GetCudaType<int> {
    static constexpr cudaDataType_t Type = CUDA_R_32I;
};



template <typename AT, typename BT, typename CT>
cublasStatus_t
rawGemm(cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        CT alpha,
        const AT *A,
        int lda,
        const BT *B,
        int ldb,
        CT beta,
        CT *C,
        int ldc) {
    auto cAT = GetCudaType<AT>::Type;
    auto cBT = GetCudaType<BT>::Type;
    auto cCt = GetCudaType<CT>::Type;

    // Always accumulate in f32
    return cublasGemmEx(handle, transa, transb, m, n, k,
                         &alpha, A, cAT, lda,
                         B, cBT, ldb,
                         &beta, C, cCt, ldc,
                         cCt, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

template <typename AT, typename BT, typename CT>
void
runMatrixMult(Tensor<CT, 2, true>& c, bool transC,
              Tensor<AT, 2, true>& a, bool transA,
              Tensor<BT, 2, true>& b, bool transB,
              CT alpha,
              CT beta,
              cublasHandle_t handle,
              cudaStream_t stream) {
    cublasSetStream(handle, stream);

    // Check that we have (m x k) * (k x n) = (m x n)
    // using the input row-major layout
    int aM = transA ? a.getSize(1) : a.getSize(0);
    int aK = transA ? a.getSize(0) : a.getSize(1);

    int bK = transB ? b.getSize(1) : b.getSize(0);
    int bN = transB ? b.getSize(0) : b.getSize(1);

    int cM = transC ? c.getSize(1) : c.getSize(0);
    int cN = transC ? c.getSize(0) : c.getSize(1);

    FAISS_ASSERT(aM == cM);
    FAISS_ASSERT(aK == bK);
    FAISS_ASSERT(bN == cN);

    FAISS_ASSERT(a.getStride(1) == 1);
    FAISS_ASSERT(b.getStride(1) == 1);
    FAISS_ASSERT(c.getStride(1) == 1);

    // Now, we have to represent the matrix multiplication in
    // column-major layout
    CT* pC = c.data();

    int m = c.getSize(1); // stride 1 size
    int n = c.getSize(0); // other size
    int k = transA ? a.getSize(0) : a.getSize(1);

    int lda = transC ? a.getStride(0) : b.getStride(0);
    int ldb = transC ? b.getStride(0) : a.getStride(0);
    int ldc = c.getStride(0);

    auto gemmTrA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto gemmTrB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    if (transC) {
        gemmTrA = transA ? CUBLAS_OP_N : CUBLAS_OP_T;
        gemmTrB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
    }

    cublasStatus_t err;

    if (transC) {
        err = rawGemm(handle,
                      gemmTrA, gemmTrB,
                      m, n, k, alpha,
                      a.data(), lda, b.data(), ldb, beta,
                      pC, ldc);
    } else {
        err = rawGemm(handle,
                      gemmTrA, gemmTrB,
                      m, n, k, alpha,
                      b.data(), lda, a.data(), ldb, beta,
                      pC, ldc);
    }

    FAISS_ASSERT_FMT(err == CUBLAS_STATUS_SUCCESS,
                     "cublas failed (%d): "
                     "(%d, %d)%s x (%d, %d)%s = (%d, %d)%s",
                     (int) err,
                     a.getSize(0), a.getSize(1), transA ? "'" : "",
                     b.getSize(0), b.getSize(1), transB ? "'" : "",
                     c.getSize(0), c.getSize(1), transC ? "'" : "");
    CUDA_TEST_ERROR();
}

template void
runMatrixMult<half, half, float>(Tensor<float, 2, true>& c, bool transC,
                                 Tensor<half, 2, true>& a, bool transA,
                                 Tensor<half, 2, true>& b, bool transB,
                                 float alpha,
                                 float beta,
                                 cublasHandle_t handle,
                                 cudaStream_t stream);
template void
runMatrixMult<float, float, float>(Tensor<float, 2, true>& c, bool transC,
                                   Tensor<float, 2, true>& a, bool transA,
                                   Tensor<float, 2, true>& b, bool transB,
                                   float alpha,
                                   float beta,
                                   cublasHandle_t handle,
                                   cudaStream_t stream);

template void
runMatrixMult<int8_t, int8_t, int>(Tensor<int, 2, true>& c, bool transC,
                                   Tensor<int8_t, 2, true>& a, bool transA,
                                   Tensor<int8_t, 2, true>& b, bool transB,
                                   int alpha,
                                   int beta,
                                   cublasHandle_t handle,
                                   cudaStream_t stream);

template <typename AT, typename BT>
void runIteratedMatrixMult(Tensor<float, 3, true>& c, bool transC,
                           Tensor<AT, 3, true>& a, bool transA,
                           Tensor<BT, 3, true>& b, bool transB,
                           float alpha,
                           float beta,
                           cublasHandle_t handle,
                           cudaStream_t stream) {
    FAISS_ASSERT(c.getSize(0) == a.getSize(0));
    FAISS_ASSERT(a.getSize(0) == b.getSize(0));

    for (int i = 0; i < a.getSize(0); ++i) {
        auto cView = c[i].view();
        auto aView = a[i].view();
        auto bView = b[i].view();

        runMatrixMult(cView, transC,
                      aView, transA,
                      bView, transB,
                      alpha, beta, handle, stream);
    }
}

template void
runIteratedMatrixMult<half, float>(Tensor<float, 3, true>& c, bool transC,
                                   Tensor<half, 3, true>& a, bool transA,
                                   Tensor<float, 3, true>& b, bool transB,
                                   float alpha,
                                   float beta,
                                   cublasHandle_t handle,
                                   cudaStream_t stream);

template void
runIteratedMatrixMult<float, float>(Tensor<float, 3, true>& c, bool transC,
                                    Tensor<float, 3, true>& a, bool transA,
                                    Tensor<float, 3, true>& b, bool transB,
                                    float alpha,
                                    float beta,
                                    cublasHandle_t handle,
                                    cudaStream_t stream);

void
runBatchMatrixMult(Tensor<float, 3, true>& c, bool transC,
                   Tensor<float, 3, true>& a, bool transA,
                   Tensor<float, 3, true>& b, bool transB,
                   float alpha,
                   float beta,
                   DeviceMemory& mem,
                   cublasHandle_t handle,
                   cudaStream_t stream) {
  FAISS_ASSERT(c.getSize(0) == a.getSize(0));
  FAISS_ASSERT(a.getSize(0) == b.getSize(0));
  cublasSetStream(handle, stream);

  // Check that we have (m x k) * (k x n) = (m x n)
  // using the input row-major layout
  int aM = transA ? a.getSize(2) : a.getSize(1);
  int aK = transA ? a.getSize(1) : a.getSize(2);

  int bK = transB ? b.getSize(2) : b.getSize(1);
  int bN = transB ? b.getSize(1) : b.getSize(2);

  int cM = transC ? c.getSize(2) : c.getSize(1);
  int cN = transC ? c.getSize(1) : c.getSize(2);

  FAISS_ASSERT(aM == cM);
  FAISS_ASSERT(aK == bK);
  FAISS_ASSERT(bN == cN);

  // Now, we have to represent the matrix multiplication in
  // column-major layout
  float* pA = transC ? a.data() : b.data();
  float* pB = transC ? b.data() : a.data();
  float* pC = c.data();

  int m = c.getSize(2); // stride 1 size
  int n = c.getSize(1); // other size
  int k = transA ? a.getSize(1) : a.getSize(2);

  int lda = transC ? a.getStride(1) : b.getStride(1);
  int ldb = transC ? b.getStride(1) : a.getStride(1);
  int ldc = c.getStride(1);

  auto gemmTrA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto gemmTrB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

  if (transC) {
    gemmTrA = transA ? CUBLAS_OP_N : CUBLAS_OP_T;
    gemmTrB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
  }

  HostTensor<float*, 1, true> hostA({a.getSize(0)});
  HostTensor<float*, 1, true> hostB({b.getSize(0)});
  HostTensor<float*, 1, true> hostC({c.getSize(0)});

  size_t aOffset = a.getStride(0);
  size_t bOffset = b.getStride(0);
  size_t cOffset = c.getStride(0);

  for (int i = 0; i < a.getSize(0); ++i) {
    hostA[i] = transC ? a.data() + i * aOffset : b.data() + i * bOffset;
    hostB[i] = transC ? b.data() + i * bOffset : a.data() + i * aOffset;
    hostC[i] = c.data() + i * cOffset;
  }

  DeviceTensor<float*, 1, true> deviceA(mem, hostA, stream);
  DeviceTensor<float*, 1, true> deviceB(mem, hostB, stream);
  DeviceTensor<float*, 1, true> deviceC(mem, hostC, stream);

  auto err =
    cublasSgemmBatched(handle,
                       gemmTrA, gemmTrB,
                       m, n, k, &alpha,
                       (const float**) deviceA.data(), lda,
                       (const float**) deviceB.data(), ldb, &beta,
                       deviceC.data(), ldc, a.getSize(0));
  FAISS_ASSERT_FMT(err == CUBLAS_STATUS_SUCCESS,
                   "cublasSgemmBatched failed (%d)", (int) err);
  CUDA_TEST_ERROR();
}

} } // namespace
