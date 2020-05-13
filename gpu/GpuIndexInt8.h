//
// Created by root on 5/7/20.
//

#ifndef FAISS_GPUINDEXINT8_H
#define FAISS_GPUINDEXINT8_H

#include <faiss/IndexInt8.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/utils/MemorySpace.h>

namespace faiss { namespace gpu {

class GpuResources;

class GpuIndexInt8 : public faiss::IndexInt8 {
public:
    GpuIndexInt8(GpuResources* resources,
                 int dims,
                 faiss::MetricType metric,
                 GpuIndexConfig config);

    inline int getDevice() const {
        return device_;
    }

    inline GpuResources* getResources() {
        return resources_;
    }

    /// Set the minimum data size for searches (in MiB) for which we use
    /// CPU -> GPU paging
    void setMinPagingSize(size_t size);

    /// Returns the current minimum data size for paged searches
    size_t getMinPagingSize() const;

    /// `x` can be resident on the CPU or any GPU; copies are performed
    /// as needed
    /// Handles paged adds if the add set is too large; calls addInternal_
    void add(faiss::Index::idx_t, const int8_t* x) override;

    /// `x` and `ids` can be resident on the CPU or any GPU; copies are
    /// performed as needed
    /// Handles paged adds if the add set is too large; calls addInternal_
    void add_with_ids(Index::idx_t n,
                      const int8_t* x,
                      const Index::idx_t* ids) override;

    /// `x`, `distances` and `labels` can be resident on the CPU or any
    /// GPU; copies are performed as needed
    void search(Index::idx_t n,
                const int8_t* x,
                Index::idx_t k,
                int* distances,
                Index::idx_t* labels) const override;

protected:
    /// Copy what we need from the CPU equivalent
    void copyFrom(const faiss::IndexInt8* index);

    /// Copy what we have to the CPU equivalent
    void copyTo(faiss::IndexInt8* index) const;

    /// Does addImpl_ require IDs? If so, and no IDs are provided, we will
    /// generate them sequentially based on the order in which the IDs are added
    virtual bool addImplRequiresIDs_() const = 0;

    /// Overridden to actually perform the add
    /// All data is guaranteed to be resident on our device
    virtual void addImpl_(int n,
                          const int8_t* x,
                          const Index::idx_t* ids) = 0;

    /// Overridden to actually perform the search
    /// All data is guaranteed to be resident on our device
    virtual void searchImpl_(int n,
                             const int8_t* x,
                             int k,
                             int* distances,
                             Index::idx_t* labels) const = 0;

private:
    /// Handles paged adds if the add set is too large, passes to
    /// addImpl_ to actually perform the add for the current page
    void addPaged_(int n,
                   const int8_t* x,
                   const Index::idx_t* ids);

    /// Calls addImpl_ for a single page of GPU-resident data
    void addPage_(int n,
                  const int8_t* x,
                  const Index::idx_t* ids);

    /// Calls searchImpl_ for a single page of GPU-resident data
    void searchNonPaged_(int n,
                         const int8_t* x,
                         int k,
                         int* outDistancesData,
                         Index::idx_t* outIndicesData) const;

    /// Calls searchImpl_ for a single page of GPU-resident data,
    /// handling paging of the data and copies from the CPU
    void searchFromCpuPaged_(int n,
                             const int8_t* x,
                             int k,
                             int* outDistancesData,
                             Index::idx_t* outIndicesData) const;

protected:
    /// Manages streams, cuBLAS handles and scratch memory for devices
    GpuResources* resources_;

    /// The GPU device we are resident on
    const int device_;

    /// The memory space of our primary storage on the GPU
    const MemorySpace memorySpace_;

    /// Size above which we page copies from the CPU to GPU
    size_t minPagedSize_;
};

}}

#endif //FAISS_GPUINDEXINT8_H
