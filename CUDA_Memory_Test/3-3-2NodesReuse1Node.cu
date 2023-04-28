/*This code tests:Memory reuse of nodes in the graph 
Can the original node 1G be reused with two subsequent 0.5G nodes?*/
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>



struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};
cudaError_t GraphPoolAttrGet(int  device, struct usageStatistics* statistics)
{
    std::cout << "-------Graph MemPool Attribute-------" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &(statistics->reserved));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetGraphMemAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemHigh, &(statistics->reservedHigh));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetGraphMemAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent, &(statistics->used));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetGraphMemAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemHigh, &(statistics->usedHigh));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetGraphMemAttribute failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "reserved is     : " << statistics->reserved << std::endl;
    std::cout << "reservedHigh is : " << statistics->reservedHigh << std::endl;
    std::cout << "used is         : " << statistics->used << std::endl;
    std::cout << "usedHigh is     : " << statistics->usedHigh << std::endl << std::endl;
    return cudaSuccess;
}

void prepareAllocParams(cudaMemAllocNodeParams* allocParams, size_t bytes,
    int device) {
    memset(allocParams, 0, sizeof(*allocParams));

    allocParams->bytesize = bytes;
    allocParams->poolProps.allocType = cudaMemAllocationTypePinned;
    allocParams->poolProps.location.id = device;
    allocParams->poolProps.location.type = cudaMemLocationTypeDevice;
}

cudaError_t test1() {
    std::cout << std::endl << "This code tests: Can the node be reused with two subsequent nodes? " << std::endl << std::endl;
    cudaError_t cudaStatus;
    float* d_a = NULL,* d_b = NULL,* d_c = NULL;
    cudaStream_t stream;
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
        return cudaErrorInvalidValue;
    }
    cudaGraph_t graph;
    cudaStatus = cudaGraphCreate(&graph,0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaGraphExec_t graphExec;
    struct usageStatistics statistics = { 0 };
    cudaMemAllocNodeParams allocParamsA, allocParamsB, allocParamsC;
    cudaGraphNode_t allocNodeA, allocNodeB, allocNodeC, freeNodeA;
    long long int size1 = 1024 * 1024 * 1024, size2 = 608 * 1024 * 1024, size3 = 416 * 1024 * 1024;
    std::cout << "allocA size is " << size1 << std::endl;
    std::cout << "allocB size is " << size2 << std::endl;
    std::cout << "allocC size is " << size3 << std::endl << std::endl;
    prepareAllocParams(&allocParamsA, 512 * 1024 * 1024, 0);
    prepareAllocParams(&allocParamsB, 608 * 1024 * 1024, 0);
    prepareAllocParams(&allocParamsC, 416 * 1024 * 1024, 0);

    cudaStatus = cudaGraphAddMemAllocNode(&allocNodeA, graph, NULL, 0, &allocParamsA);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphAddMemAllocNode failed!");
        return cudaErrorInvalidValue;
    }
    d_a = (float*)allocParamsA.dptr;
    cudaStatus = cudaGraphAddMemFreeNode(&freeNodeA, graph, &allocNodeA, 1, (void*)d_a);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphAddMemFreeNode failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphAddMemAllocNode(&allocNodeB, graph, &freeNodeA, 1, &allocParamsB);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphAddMemAllocNode failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphAddMemAllocNode(&allocNodeC, graph, &freeNodeA, 1, &allocParamsC);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphAddMemAllocNode failed!");
        return cudaErrorInvalidValue;
    }
    d_b = (float*)allocParamsB.dptr;
    d_c = (float*)allocParamsC.dptr;

    std::cout << "allocA address is " << d_a << std::endl;
    std::cout << "allocB address is " << d_b << std::endl;
    std::cout << "allocC address is " << d_c << std::endl << std::endl;
    cudaStatus = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphLaunch(graphExec, stream);//launch graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = GraphPoolAttrGet(0, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GraphPoolAttrGet failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphDestroy(graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphExecDestroy(graphExec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphExecDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}


int main(int argc, char** argv)
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

    cudaStatus = test1();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test1 failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

