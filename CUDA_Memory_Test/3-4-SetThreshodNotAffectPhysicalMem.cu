/*This code tests: Whether the physical memory changes when the threshold is set for the memory pool*/
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};

cudaError_t poolAttrGet(cudaMemPool_t memPool, struct usageStatistics* statistics)
{
    std::cout << "-------MemPool Attribute-------" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent,&(statistics->reserved));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolGetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &(statistics->reservedHigh));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolGetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, &(statistics->used));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolGetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &(statistics->usedHigh));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolGetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "reserved is     : " << statistics->reserved << std::endl;
    std::cout << "reservedHigh is : " << statistics->reservedHigh << std::endl;
    std::cout << "used is         : " << statistics->used << std::endl;
    std::cout << "usedHigh is     : " << statistics->usedHigh << std::endl << std::endl;
    return cudaSuccess;
}


cudaError_t test1() {
    std::cout << std::endl<<"This code tests:" << " Whether the physical memory changes when the threshold is set for the memory pool" << std::endl << std::endl;
    int device = 0;
    cudaError_t cudaStatus;

    struct usageStatistics statistics = { 0,0,0,0 };

    cudaMemPoolProps poolProps = { };//set pool properties
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = 0;
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.handleTypes = cudaMemHandleTypeNone;
    cudaMemPool_t memPool;
    cudaStream_t stream;
    cudaStatus=cudaMemPoolCreate(&memPool, &poolProps);//create explicit pool
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaDeviceSetMemPool(device, memPool);//set explicit pool as current pool
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetMemPool failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "before set threshold " << std::endl;
    cudaStatus = poolAttrGet(memPool, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "poolAttrGet failed!");
        return cudaErrorInvalidValue;
    }
    unsigned int setVal = 1 << 30;//set threshold
    cudaStatus = cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolSetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "after set threshold " << std::endl;
    cudaStatus = poolAttrGet(memPool, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "poolAttrGet failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaMemPoolDestroy(memPool);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    return cudaStatus;
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

    cudaStatus=cudaDriverGetVersion(&driverVersion);
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

    cudaStatus = test1();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test1 failed!");
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
