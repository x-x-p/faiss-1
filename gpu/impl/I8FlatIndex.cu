//
// Created by root on 5/7/20.
//

#include <faiss/gpu/impl/I8FlatIndex.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Transpose.cuh>

namespace faiss { namespace gpu {

I8FlatIndex::I8FlatIndex(GpuResources* res,
                         int dim,
                         bool storeTransposed,
                         MemorySpace space) :
        resources_(res),
        dim_(dim),
        storeTransposed_(storeTransposed),
        space_(space),
        num_(0),
        rawData_(space) {}

int I8FlatIndex::getSize() const {
    return vectors_.getSize(0);
}

int I8FlatIndex::getDim() const {
    return vectors_.getSize(1);
}

void
I8FlatIndex::reserve(size_t numVecs, cudaStream_t stream) {
    rawData_.reserve(numVecs * dim_, stream);
}


Tensor<int8_t, 2, true>&
I8FlatIndex::getVectorsRef() {
    // Should not call this unless we are in flreturn vectors_;
    return vectors_;
}

void
I8FlatIndex::query(Tensor<int8_t, 2, true>& input,
                 int k,
                 faiss::MetricType metric,
                 Tensor<int, 2, true>& outDistances,
                 Tensor<int, 2, true>& outIndices,
                 bool exactDistance) {
    bfKnnOnDevice(resources_,
                  getCurrentDevice(),
                  resources_->getDefaultStreamCurrentDevice(),
                  storeTransposed_ ? vectorsTransposed_ : vectors_,
                  !storeTransposed_, // is vectors row major?
                  &norms_,
                  input,
                  true, // input is row major
                  k,
                  metric,
                  0,
                  outDistances,
                  outIndices,
                  !exactDistance);
}

void
I8FlatIndex::reconstruct(Tensor<int, 1, true> &listIds,
                         Tensor<int8_t, 2, true> &vecs) {
    runReconstruct(listIds,
                   getVectorsRef(),
                   vecs,
                   resources_->getDefaultStreamCurrentDevice());
}

void
I8FlatIndex::reconstruct(Tensor<int, 2, true> &listIds,
                         Tensor<int8_t, 3, true> &vecs) {
    auto listIds1 = listIds.downcastOuter<1>();
    auto vecs2 = vecs.downcastOuter<2>();

    reconstruct(listIds1, vecs2);
}

void
I8FlatIndex::add(const int8_t* data, int numVecs, cudaStream_t stream) {
    if (numVecs == 0) {
        return;
    }

    rawData_.append((char*) data,
                    (size_t) dim_ * numVecs,
                    stream,
                    true /* reserve exactly */);

    num_ += numVecs;

    DeviceTensor<int8_t, 2, true> vectors(
            (int8_t*) rawData_.data(), {(int) num_, dim_}, space_);
    vectors_ = std::move(vectors);

    if (storeTransposed_) {
        vectorsTransposed_ =
                std::move(DeviceTensor<int8_t, 2, true>({dim_, (int) num_}, space_));
        runTransposeAny(vectors_, 0, 1, vectorsTransposed_, stream);
    }
}


// todo: impl bulk remove
void
I8FlatIndex::del(int id, cudaStream_t stream) {
    if(id > num_ - 1) {
        return;
    }

    if(id < num_ - 1){
        //不释放以前申请的
        CUDA_VERIFY(cudaMemcpy(
                ((char*)rawData_.data()) + id * dim_,
                ((char*)rawData_.data()) + (num_ - 1) * dim_,
                dim_, //In bytes
                cudaMemcpyDeviceToDevice
        ));
    }

    num_ -= 1;
    rawData_.resize(num_ * dim_, stream);

    {
        DeviceTensor<int8_t, 2, true> vectors(
                (int8_t*) rawData_.data(), {(int) num_, dim_}, space_);
        vectors_ = std::move(vectors);
    }

    if (storeTransposed_) {
        {
            vectorsTransposed_ =
                    std::move(DeviceTensor<int8_t, 2, true>({dim_, (int) num_}, space_));
            runTransposeAny(vectors_, 0, 1, vectorsTransposed_, stream);
        }
    }
}

void I8FlatIndex::reset() {
    rawData_.clear();
    vectors_ = std::move(DeviceTensor<int8_t , 2, true>());
    vectorsTransposed_ = std::move(DeviceTensor<int8_t, 2, true>());
    norms_ = std::move(DeviceTensor<int, 1, true>());
    num_ = 0;
}


}}