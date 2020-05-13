//
// Created by root on 5/7/20.
//

#ifndef FAISS_INDEXINT8FLAT_H
#define FAISS_INDEXINT8FLAT_H

#include <vector>

#include <faiss/IndexInt8.h>


namespace faiss {

struct IndexInt8Flat : public IndexInt8 {
    /// database vectors, size ntotal * d
    std::vector<int8_t> xb;

    explicit IndexInt8Flat (idx_t d, MetricType metric = METRIC_INNER_PRODUCT);

    void add(idx_t n, const int8_t* x) override;

    void reset() override;

    void search(
            idx_t n,
            const int8_t* x,
            idx_t k,
            int* distances,
            idx_t* labels) const override;

    void reconstruct(idx_t i, int8_t* recons) const override;

    /** compute distance with a subset of vectors
     *
     * @param x       query vectors, size n * d
     * @param labels  indices of the vectors that should be compared
     *                for each query vector, size n * k
     * @param distances
     *                corresponding output distances, size n * k
     */
    void compute_distance_subset (
            idx_t n,
            const int8_t *x,
            idx_t k,
            int *distances,
            const idx_t *labels) const;

    /** remove some id. NB that Because of the structure of the
     * indexing structure, the semantics of this operation are
     * different from the usual ones: pos i will exchange ntotal - 1 */
    size_t remove_id(idx_t i) override;

    void update(idx_t i, const int8_t* x) override ;
};

}


#endif //FAISS_INDEXINT8FLAT_H
