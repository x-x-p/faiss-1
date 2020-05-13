//
// Created by root on 5/7/20.
//

#ifndef FAISS_I8FLATINDEX_H
#define FAISS_I8FLATINDEX_H

#include <faiss/MetricType.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <faiss/gpu/utils/MemorySpace.h>

namespace faiss { namespace gpu {

class GpuResources;

class I8FlatIndex {
public:
    I8FlatIndex(GpuResources* res,
                int dim,
                bool storeTransposed,
                MemorySpace space);

    /// Returns the number of vectors we contain
    int getSize() const;

    /// Returns the dimensionality of the vectors
    int getDim() const;

    /// Reserve storage that can contain at least this many vectors
    void reserve(size_t numVecs, cudaStream_t stream);

    /// Returns the vectors
    Tensor<int8_t, 2, true>& getVectorsRef();

    void query(Tensor<int8_t, 2, true>& vecs,
               int k,
               faiss::MetricType metric,
               Tensor<int, 2, true>& outDistances,
               Tensor<int, 2, true>& outIndices,
               bool exactDistance);

    /// Gather vectors given the set of IDs
    void reconstruct(Tensor<int, 1, true>& listIds,
                     Tensor<int8_t, 2, true>& vecs);

    void reconstruct(Tensor<int, 2, true>& listIds,
                     Tensor<int8_t, 3, true>& vecs);

    /// Add vectors to ourselves; the pointer passed can be on the host
    /// or the device
    void add(const int8_t* data, int numVecs, cudaStream_t stream);

    void del(int key, cudaStream_t stream);

    /// Free all storage
    void reset();

private:
    /// Collection of GPU resources that we use
    GpuResources* resources_;

    /// Dimensionality of our vectors
    const int dim_;

    /// Store vectors in transposed layout for speed; makes addition to
    /// the index slower
    const bool storeTransposed_;

    /// Memory space for our allocations
    MemorySpace space_;

    /// How many vectors we have
    int num_;

    /// The underlying expandable storage
    DeviceVector<char> rawData_;

    /// Vectors currently in rawData_
    DeviceTensor<int8_t , 2, true> vectors_;
    DeviceTensor<int8_t, 2, true> vectorsTransposed_;

    /// Precomputed L2 norms
    DeviceTensor<int32_t , 1, true> norms_;
};

}}

#endif //FAISS_I8FLATINDEX_H
