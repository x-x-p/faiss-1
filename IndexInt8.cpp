//
// Created by root on 5/7/20.
//

#include <faiss/IndexInt8.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <cstring>

namespace faiss {

IndexInt8::~IndexInt8 () = default;


void IndexInt8::train(long n, const int8_t *x) {
// does nothing by default
}

void IndexInt8::add_with_ids(idx_t n,
                             const int8_t *x,
                             const idx_t *xids) {
    FAISS_THROW_MSG ("add_with_ids not implemented for this type of index");
}


void IndexInt8::range_search(idx_t n, const int8_t *x, int radius,
                             RangeSearchResult *result) const
{
    FAISS_THROW_MSG ("range search not implemented");
}

void IndexInt8::assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k)
{
    auto * distances = new distance_t[n * k];
    ScopeDeleter<distance_t> del(distances);
    search (n, x, k, distances, labels);
}

size_t IndexInt8::remove_ids(const IDSelector& /*sel*/) {
    FAISS_THROW_MSG ("remove_ids not implemented for this type of index");
    return -1;
}

size_t IndexInt8::remove_id(faiss::Index::idx_t /*i*/) {
    FAISS_THROW_MSG ("remove_ids not implemented for this type of index");
    return -1;
}

void IndexInt8::update(faiss::Index::idx_t i, const int8_t *recons){
    FAISS_THROW_MSG ("update not implemented for this type of index");
}

void IndexInt8::reconstruct (idx_t, int8_t * ) const {
    FAISS_THROW_MSG ("reconstruct not implemented for this type of index");
}


void IndexInt8::reconstruct_n (idx_t i0, idx_t ni, int8_t *recons) const {
    for (idx_t i = 0; i < ni; i++) {
        reconstruct (i0 + i, recons + i * d);
    }
}


void IndexInt8::search_and_reconstruct (idx_t n, const int8_t *x, idx_t k,
                                        int32_t *distances, idx_t *labels,
                                        int8_t *recons) const {
    search (n, x, k, distances, labels);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = 0; j < k; ++j) {
            idx_t ij = i * k + j;
            idx_t key = labels[ij];
            auto* reconstructed = recons + ij * d;
            if (key < 0) {
                // Fill with NaNs
                memset(reconstructed, -1, sizeof(*reconstructed) * d);
            } else {
                reconstruct (key, reconstructed);
            }
        }
    }
}
}