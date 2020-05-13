//
// Created by root on 5/7/20.
//
#ifndef FAISS_GPUINDEXINT8FLAT_H
#define FAISS_GPUINDEXINT8FLAT_H

#include <faiss/gpu/GpuIndexInt8.h>

namespace faiss {
struct IndexInt8Flat;
}

namespace faiss { namespace gpu {

class I8FlatIndex;

struct GpuIndexInt8FlatConfig : public GpuIndexConfig {
    inline GpuIndexInt8FlatConfig()
            : storeTransposed(false) {
    }

    /// Whether or not data is stored (transparently) in a transposed
    /// layout, enabling use of the NN GEMM call, which is ~10% faster.
    /// This will improve the speed of the flat index, but will
    /// substantially slow down any add() calls made, as all data must
    /// be transposed, and will increase storage requirements (we store
    /// data in both transposed and non-transposed layouts).
    bool storeTransposed;
};

class GpuIndexInt8Flat : public GpuIndexInt8 {
public:
    /// Construct from a pre-existing faiss::IndexFlat instance, copying
    /// data over to the given GPU
    GpuIndexInt8Flat(GpuResources *resources,
                     const faiss::IndexInt8Flat *index,
                     GpuIndexInt8FlatConfig config = GpuIndexInt8FlatConfig());

    /// Construct an empty instance that can be added to
    GpuIndexInt8Flat(GpuResources *resources,
                     int dims,
                     GpuIndexInt8FlatConfig config = GpuIndexInt8FlatConfig());

    ~GpuIndexInt8Flat() override;

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss::IndexInt8Flat *index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexInt8Flat *index) const;

    /// Returns the number of vectors we contain
    size_t getNumVecs() const;

    /// Clears all vectors from this index
    void reset() override;

    /// This index is not trained, so this does nothing
    void train(Index::idx_t n, const int8_t *x) override;

    /// Overrides to avoid excessive copies
    void add(Index::idx_t, const int8_t *x) override;

    size_t remove_id(Index::idx_t i) override;

    void update(Index::idx_t i, const int8_t *data) override;

    /// Reconstruction methods; prefer the batch reconstruct as it will
    /// be more efficient
    void reconstruct(faiss::Index::idx_t key, int8_t *out) const override;

    /// Batch reconstruction method
    void reconstruct_n(faiss::Index::idx_t i0,
                       faiss::Index::idx_t num,
                       int8_t *out) const override;

    /// For internal access
    inline I8FlatIndex *getGpuData() { return data_; }

protected:
    /// Flat index does not require IDs as there is no storage available for them
    bool addImplRequiresIDs_() const override;

    /// Called from GpuIndex for add
    void addImpl_(int n,
                  const int8_t *x,
                  const Index::idx_t *ids) override;

    /// Called from GpuIndex for search
    void searchImpl_(int n,
                     const int8_t *x,
                     int k,
                     int *distances,
                     faiss::Index::idx_t *labels) const override;

protected:
    /// Our config object
    const GpuIndexInt8FlatConfig config_;

    /// Holds our GPU data containing the list of vectors; is managed via raw
    /// pointer so as to allow non-CUDA compilers to see this header
    I8FlatIndex *data_;
};

}}

#endif //FAISS_GPUINDEXINT8FLAT_H
