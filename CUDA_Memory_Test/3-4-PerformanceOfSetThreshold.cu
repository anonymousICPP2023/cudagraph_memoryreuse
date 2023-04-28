/*This code tests:
compare performance differences of cudaMallocAsync between setting the threshold and do not set the threshold. */

#include <cuda_runtime.h>
#include<cuda.h>
#include <iostream>

#define LOOPTIMES 1000 //loop times
#define SIZE 32 * 1024 * 1024 //size of allocation

cudaError_t test1() {
    int device = 0;// Choose which GPU to run on, change this on a multi-GPU system.
    std::cout << std::endl << "This code tests:  performance differences of cudaMallocAsync between setting the threshold and do not set the threshold." << std::endl << std::endl;
    std::cout << "set threshold---cudaMallocAsync+cudaFreeAsync---Synchronize in each loop" << std::endl << std::endl;
    cudaError_t cudaStatus ;
    int* d_a = NULL;
    cudaMemPoolProps poolProps = { };//set pool properties
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = 0;
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.handleTypes = cudaMemHandleTypeNone;
    cudaMemPool_t memPool;
    cudaStream_t stream;
    cudaStatus = cudaMemPoolCreate(&memPool, &poolProps);//create explicit pool
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolCreate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaDeviceSetMemPool(device, memPool);//set explicit pool as current pool
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetMemPool failed!");
        return cudaErrorInvalidValue;
    }

    unsigned long long int setVal = UINT64_MAX;//set threshold
    cudaStatus = cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolSetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    float time = 0.0f;
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }


    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaMallocAsync((void**)&d_a, SIZE, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocAsync failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaFreeAsync(d_a, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaFreeAsync failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaStreamSynchronize(stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaStreamSynchronize failed!");
            return cudaErrorInvalidValue;
        }
    }

    cudaStatus = cudaEventRecord(stop, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventSynchronize failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventElapsedTime(&time, start, stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventElapsedTime failed!");
        return cudaErrorInvalidValue;
    }
    printf("test1 time is %f\n", time);

    cudaStatus = cudaEventDestroy(start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventDestroy(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
cudaError_t test2() {
    std::cout << std::endl << "This code tests:  performance differences of cudaMallocAsync between setting the threshold and do not set the threshold." << std::endl << std::endl;
    std::cout << "no threshold---cudaMallocAsync+cudaFreeAsync---Synchronize in each loop" << std::endl << std::endl;
    cudaError_t cudaStatus;
    int* d_a = NULL;
    cudaStream_t stream;
    cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed!");
        return cudaErrorInvalidValue;
    }
    float time = 0.0f;
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaMallocAsync((void**)&d_a, SIZE, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocAsync failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaFreeAsync(d_a, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaFreeAsync failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaStreamSynchronize(stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaStreamSynchronize failed!");
            return cudaErrorInvalidValue;
        }
    }

    cudaStatus = cudaEventRecord(stop, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventSynchronize failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventElapsedTime(&time, start, stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventElapsedTime failed!");
        return cudaErrorInvalidValue;
    }
    printf("test2 time is %f\n", time);

    cudaStatus = cudaEventDestroy(start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventDestroy(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
int main()
{
    cudaError_t cudaStatus;
    int device = 0;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    int driverVersion = 0;
    int deviceSupportsMemoryPools = 0;

    cudaStatus = cudaDriverGetVersion(&driverVersion);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDriverGetVersion failed!");
        return 1;
    }
    printf("Driver version is: %d.%d\n", driverVersion / 1000,
        (driverVersion % 100) / 10);

    if (driverVersion < 11040) {
        printf("Waiving execution as driver does not support Graph Memory Nodes\n");
        return 1;
    }

    cudaStatus = cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
        cudaDevAttrMemoryPoolsSupported, device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed!");
        return 1;
    }
    if (!deviceSupportsMemoryPools) {
        printf("Waiving execution as device does not support Memory Pools\n");
        return 1;
    }
    else {
        printf("Running sample.\n");
    }

     cudaStatus = test1();//set threshold---cudaMallocAsync+cudaFreeAsync---Synchronize in each loop
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test1 failed!");
        return 1;
    }
    //  cudaStatus = test2();//no threshold---cudaMallocAsync+cudaFreeAsync---Synchronize in each loop
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test2 failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

