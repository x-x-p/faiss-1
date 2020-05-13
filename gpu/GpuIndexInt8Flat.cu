//
// Created by root on 5/7/20.
//

#include <faiss/gpu/GpuIndexInt8Flat.h>
#include <faiss/IndexInt8Flat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/I8FlatIndex.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <limits>

namespace faiss { namespace gpu {


GpuIndexInt8Flat::GpuIndexInt8Flat(GpuResources *resources, const faiss::IndexInt8Flat *index,
                                   GpuIndexInt8FlatConfig config) :
        GpuIndexInt8(resources,
                     index->d,
                     index->metric_type,
                     config),
        config_(std::move(config)),
        data_(nullptr) {
    // Flat index doesn't need training
    this->is_trained = true;
    copyFrom(index);
}

GpuIndexInt8Flat::GpuIndexInt8Flat(GpuResources *resources, int dims,
                                   GpuIndexInt8FlatConfig config) :
        GpuIndexInt8(resources, dims, METRIC_INNER_PRODUCT, config),
        config_(std::move(config)),
        data_(nullptr) {
    // Flat index doesn't need training
    this->is_trained = true;

    // Construct index
    DeviceScope scope(device_);
    data_ = new I8FlatIndex(resources,
                            dims,
                            config_.storeTransposed,
                            memorySpace_);
}

GpuIndexInt8Flat::~GpuIndexInt8Flat() {
    delete data_;
}

void
GpuIndexInt8Flat::copyFrom(const faiss::IndexInt8Flat *index)
{
    // todo impl
}

void GpuIndexInt8Flat::copyTo(faiss::IndexInt8Flat *index) const {
    // todo impl
}

size_t
GpuIndexInt8Flat::getNumVecs() const {
    return this->ntotal;
}

void
GpuIndexInt8Flat::reset() {
    DeviceScope scope(device_);

    // Free the underlying memory
    data_->reset();
    this->ntotal = 0;
}

void
GpuIndexInt8Flat::train(Index::idx_t n, const int8_t* x) {
    // nothing to do
}

size_t
GpuIndexInt8Flat::remove_id(faiss::Index::idx_t i)
{
    if(i > ntotal - 1) {
        return 0;
    }
    DeviceScope scope(device_);
    data_->del(i, resources_->getDefaultStream(device_));
    return 1;
}

void
GpuIndexInt8Flat::update(idx_t key, const int8_t *recons)
{
    if(key > ntotal - 1) {
        return;
    }
    DeviceScope scope(device_);
    auto stream = resources_->getDefaultStream(device_);

    auto vec = data_->getVectorsRef()[key];
    toDevice(vec.data(), recons, d, stream);
}

void
GpuIndexInt8Flat::add(Index::idx_t n, const int8_t* x) {
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(n <= (Index::idx_t) std::numeric_limits<int>::max(),
                           "GPU index only supports up to %d indices",
                           std::numeric_limits<int>::max());

    if (n == 0) {
        // nothing to add
        return;
    }

    DeviceScope scope(device_);

    // To avoid multiple re-allocations, ensure we have enough storage
    // available
    data_->reserve(n, resources_->getDefaultStream(device_));

    // If we're not operating in float16 mode, we don't need the input
    // data to be resident on our device; we can add directly.
    addImpl_(n, x, nullptr);
}

bool
GpuIndexInt8Flat::addImplRequiresIDs_() const {
    return false;
}

void
GpuIndexInt8Flat::addImpl_(int n,
                           const int8_t* x,
                           const Index::idx_t* ids) {
    FAISS_ASSERT(data_);
    FAISS_ASSERT(n > 0);

    // We do not support add_with_ids
    FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");

    // Due to GPU indexing in int32, we can't store more than this
    // number of vectors on a GPU
    FAISS_THROW_IF_NOT_FMT(this->ntotal + n <=
                           (faiss::Index::idx_t) std::numeric_limits<int>::max(),
                           "GPU index only supports up to %zu indices",
                           (size_t) std::numeric_limits<int>::max());

    data_->add(x, n, resources_->getDefaultStream(device_));
    this->ntotal += n;
}

void
GpuIndexInt8Flat::searchImpl_(int n,
                              const int8_t* x,
                              int k,
                              int* distances,
                              Index::idx_t* labels) const {
    auto stream = resources_->getDefaultStream(device_);

    // Input and output data are already resident on the GPU
    Tensor<int8_t, 2, true> queries(const_cast<int8_t *>(x), {n, (int) this->d});
    Tensor<int, 2, true> outDistances(distances, {n, k});
    Tensor<Index::idx_t, 2, true> outLabels(labels, {n, k});

    // FlatIndex only supports int indices
    DeviceTensor<int, 2, true> outIntLabels(
            resources_->getMemoryManagerCurrentDevice(), {n, k}, stream);

    data_->query(queries, k, metric_type,
                 outDistances, outIntLabels, true);

    // Convert int to idx_t
    convertTensor<int, faiss::Index::idx_t, 2>(stream,
                                               outIntLabels,
                                               outLabels);
}

void
GpuIndexInt8Flat::reconstruct(faiss::Index::idx_t key,
                              int8_t* out) const {
    DeviceScope scope(device_);

    FAISS_THROW_IF_NOT_MSG(key < this->ntotal, "index out of bounds");
    auto stream = resources_->getDefaultStream(device_);

    auto vec = data_->getVectorsRef()[key];
    fromDevice(vec.data(), out, this->d, stream);
}

void
GpuIndexInt8Flat::reconstruct_n(faiss::Index::idx_t i0,
                                faiss::Index::idx_t num,
                                int8_t* out) const {
    DeviceScope scope(device_);

    FAISS_THROW_IF_NOT_MSG(i0 < this->ntotal, "index out of bounds");
    FAISS_THROW_IF_NOT_MSG(i0 + num - 1 < this->ntotal, "num out of bounds");
    auto stream = resources_->getDefaultStream(device_);

    auto vec = data_->getVectorsRef()[i0];
    fromDevice(vec.data(), out, this->d * num, stream);
}

}}