//
// Created by root on 5/8/20.
//

#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>

namespace faiss { namespace gpu {

// warp Q to thread Q:
// 1, 1
// 32, 2
// 64, 3
// 128, 3
// 256, 4
// 512, 8
// 1024, 8
// 2048, 8

BLOCK_SELECT_DECL(int, true, 1);
BLOCK_SELECT_DECL(int, true, 32);
BLOCK_SELECT_DECL(int, true, 64);
BLOCK_SELECT_DECL(int, true, 128);
BLOCK_SELECT_DECL(int, true, 256);
BLOCK_SELECT_DECL(int, true, 512);
BLOCK_SELECT_DECL(int, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_DECL(int, true, 2048);
#endif

BLOCK_SELECT_DECL(int, false, 1);
BLOCK_SELECT_DECL(int, false, 32);
BLOCK_SELECT_DECL(int, false, 64);
BLOCK_SELECT_DECL(int, false, 128);
BLOCK_SELECT_DECL(int, false, 256);
BLOCK_SELECT_DECL(int, false, 512);
BLOCK_SELECT_DECL(int, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_DECL(int, false, 2048);
#endif

void runBlockSelect(Tensor<int, 2, true>& in,
                    Tensor<int, 2, true>& outK,
                    Tensor<int, 2, true>& outV,
                    bool dir, int k, cudaStream_t stream) {
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (dir) {
        if (k == 1) {
            BLOCK_SELECT_CALL(int, true, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_CALL(int, true, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_CALL(int, true, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_CALL(int, true, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_CALL(int, true, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_CALL(int, true, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_CALL(int, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_CALL(int, true, 2048);
#endif
        }
    } else {
        if (k == 1) {
            BLOCK_SELECT_CALL(int, false, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_CALL(int, false, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_CALL(int, false, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_CALL(int, false, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_CALL(int, false, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_CALL(int, false, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_CALL(int, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_CALL(int, false, 2048);
#endif
        }
    }
}

void runBlockSelectPair(Tensor<int, 2, true>& inK,
                        Tensor<int, 2, true>& inV,
                        Tensor<int, 2, true>& outK,
                        Tensor<int, 2, true>& outV,
                        bool dir, int k, cudaStream_t stream) {
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (dir) {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(int, true, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(int, true, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(int, true, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(int, true, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(int, true, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(int, true, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(int, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(int, true, 2048);
#endif
        }
    } else {
        if (k == 1) {
            BLOCK_SELECT_PAIR_CALL(int, false, 1);
        } else if (k <= 32) {
            BLOCK_SELECT_PAIR_CALL(int, false, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_PAIR_CALL(int, false, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_PAIR_CALL(int, false, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_PAIR_CALL(int, false, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_PAIR_CALL(int, false, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_PAIR_CALL(int, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_PAIR_CALL(int, false, 2048);
#endif
        }
    }
}

} } // namespace