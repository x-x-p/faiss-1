//
// Created by root on 5/9/20.
//

#include <cstdio>
#include <cstdlib>

#include <sys/types.h>
#include <exception>

#include <cuda_profiler_api.h>
#include <faiss/gpu/GpuIndexInt8Flat.h>
#include <faiss/gpu/StandardGpuResources.h>

int main() {
    int d = 384;                            // dimension
    int nb = 200000;                       // database size
    int nq = 1024;                        // nb of queries

    int8_t *xb = new int8_t[d * nb];
    int8_t *xq = new int8_t[d * nq];

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }

    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexInt8FlatConfig config;
    config.storeTransposed = true;
    faiss::gpu::GpuIndexInt8Flat index(&res, d, config);           // call constructor
    index.add(nb, xb);                     // add vectors to the index
    printf("ntotal = %ld\n", index.ntotal);


    int k = 256;
    {       // search xq
        long *I = new long[k * nq];
        int *D = new int[k * nq];

        cudaProfilerStart();
        index.search(nq, xq, k, D, I);
        cudaProfilerStop();

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < 5; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < 5; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }



    delete [] xb;
    delete [] xq;

    return 0;
}