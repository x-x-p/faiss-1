//
// Created by root on 5/7/20.
//

#ifndef FAISS_INDEXINT8_H
#define FAISS_INDEXINT8_H

#include <faiss/MetricType.h>
#include <faiss/Index.h>
#include <cstdio>
#include <typeinfo>
#include <string>
#include <sstream>

namespace faiss {

struct IndexInt8 {
    using idx_t = Index::idx_t;    ///< all indices are this type
    using component_t = uint8_t;
    using distance_t = int32_t;

    int d;                 ///< vector dimension
    idx_t ntotal;          ///< total nb of indexed vectors
    bool verbose;          ///< verbosity level

    /// set if the Index does not require training, or if training is done already
    bool is_trained;

    /// type of metric this index uses for search
    MetricType metric_type;

    explicit IndexInt8(idx_t d = 0, MetricType metric = METRIC_INNER_PRODUCT)
            : d(d),
              ntotal(0),
              verbose(false),
              is_trained(true),
              metric_type(metric) {}

    virtual ~IndexInt8();


    /** Perform training on a representative set of vectors.
     *
     * @param n      nb of training vectors
     * @param x      training vecors
     */
    virtual void train(idx_t n, const int8_t *x);

    /** Add n vectors of dimension d to the index.
     *
     * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
     * @param x      input matrix
     */
    virtual void add(idx_t n, const int8_t *x) = 0;

    /** Same as add, but stores xids instead of sequential ids.
     *
     * The default implementation fails with an assertion, as it is
     * not supported by all indexes.
     *
     * @param xids if non-null, ids to store for the vectors (size n)
     */
    virtual void add_with_ids(idx_t n, const int8_t *x, const idx_t *xids);

    /** Query n vectors of dimension d to the index.
     *
     * return at most k vectors. If there are not enough results for a
     * query, the result array is padded with -1s.
     *
     * @param x           input vectors to search, size n * d
     * @param labels      output labels of the NNs, size n*k
     * @param distances   output pairwise distances, size n*k
     */
    virtual void search(idx_t n, const int8_t *x, idx_t k,
                        int32_t *distances, idx_t *labels) const = 0;

    /** Query n vectors of dimension d to the index.
     *
     * return all vectors with distance < radius. Note that many indexes
     * do not implement the range_search (only the k-NN search is
     * mandatory). The distances are converted to float to reuse the
     * RangeSearchResult structure, but they are integer. By convention,
     * only distances < radius (strict comparison) are returned,
     * ie. radius = 0 does not return any result and 1 returns only
     * exact same vectors.
     *
     * @param x           input vectors to search, size n * d
     * @param radius      search radius
     * @param result      result table
     */
    virtual void range_search(idx_t n, const int8_t *x, int radius,
                              RangeSearchResult *result) const;

    /** Return the indexes of the k vectors closest to the query x.
     *
     * This function is identical to search but only returns labels of neighbors.
     * @param x           input vectors to search, size n * d
     * @param labels      output labels of the NNs, size n*k
     */
    void assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k = 1);

    /// Removes all elements from the database.
    virtual void reset() = 0;

    /** Removes IDs from the index. Not supported by all indexes.
     */
    virtual size_t remove_ids(const IDSelector &sel);

    /** removes ID from the index. Not supported by all
     * indexes. Returns the number of elements removed.
     */
    virtual size_t remove_id(idx_t i);

    /** update id to the index. Not support by all
     *  indexes.
     */
    virtual void update(idx_t i, const int8_t *recons);

    /** Reconstruct a stored vector.
     *
     * This function may not be defined for some indexes.
     * @param key         id of the vector to reconstruct
     * @param recons      reconstucted vector (size d)
     */
    virtual void reconstruct(idx_t key, int8_t *recons) const;


    /** Reconstruct vectors i0 to i0 + ni - 1.
     *
     * This function may not be defined for some indexes.
     * @param recons      reconstucted vectors (size ni * d)
     */
    virtual void reconstruct_n(idx_t i0, idx_t ni, int8_t *recons) const;

    /** Similar to search, but also reconstructs the stored vectors
     * for the search results.
     *
     * If there are not enough results for a query, the resulting array
     * is padded with -1s.
     *
     * @param recons      reconstructed vectors size (n, k, d)
     **/
    virtual void search_and_reconstruct(idx_t n, const int8_t *x, idx_t k,
                                        int32_t *distances, idx_t *labels,
                                        int8_t *recons) const;
};

}

#endif //FAISS_INDEXINT8_H
