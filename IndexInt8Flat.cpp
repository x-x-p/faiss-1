//
// Created by root on 5/7/20.
//

#include <faiss/IndexInt8Flat.h>

#include <cstring>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/AuxIndexStructures.h>


namespace faiss {

IndexInt8Flat::IndexInt8Flat(idx_t d, MetricType metric) :
        IndexInt8(d, metric) { }

void
IndexInt8Flat::add(idx_t n, const int8_t *x) {
    xb.insert(xb.end(), x, x + n * d);
    ntotal += n;
}

void
IndexInt8Flat::reset() {
    xb.clear();
    ntotal = 0;
}

void
IndexInt8Flat::search(idx_t n, const int8_t *x, idx_t k, int *distances,
                      idx_t *labels) const {
    FAISS_ASSERT(metric_type == METRIC_INNER_PRODUCT);

    int_minheap_array_t res = {
            size_t(n), size_t(k), labels, distances};

    knn_inner_product(x, xb.data(), d, n, ntotal, &res);
}

void IndexInt8Flat::reconstruct(IndexInt8::idx_t i, int8_t *recons) const {
    IndexInt8::reconstruct(i, recons);
}

void IndexInt8Flat::compute_distance_subset(IndexInt8::idx_t n,
                                            const int8_t *x, IndexInt8::idx_t k, int *distances,
                                            const IndexInt8::idx_t *labels) const {

}

size_t IndexInt8Flat::remove_id(IndexInt8::idx_t i) {
    FAISS_THROW_IF_NOT(i >= 0 && i < ntotal);

    std::memcpy(&xb[i * d], &xb[(ntotal-1) * d], d);
    ntotal -= 1;
}

void IndexInt8Flat::update(IndexInt8::idx_t i, const int8_t *x) {
    FAISS_THROW_IF_NOT(i >= 0 && i < ntotal);

    std::memcpy(&xb[i * d], x, d);
}

}