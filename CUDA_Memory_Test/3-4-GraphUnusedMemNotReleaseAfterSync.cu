/*This code tests:Graph memory pool does not release memory after synchronization*/
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

cudaError_t test() {
    std::cout << std::endl << "This code tests:Graph memory pool does not release memory after synchronization" << std::endl << std::endl;
    cudaError_t cudaStatus;
    struct usageStatistics statistics= { 0,0,0,0 };
    cudaGraphExec_t graphExec;
    cudaGraph_t graph;
    cudaStatus = cudaGraphCreate(&graph,0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream;
    int* d_a = NULL;


    int SIZE = 1 * 1024 * 1024;

    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed!");
        return cudaErrorInvalidValue;
    }
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
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }

    std::cout << std::endl << "------before instantiate-------" << std::endl;
    std::cout << "d_a is " << d_a << std::endl;
    cudaStatus = GraphPoolAttrGet(0, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GraphPoolAttrGet failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);//instantiate graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }


    std::cout << std::endl << "------before launch-------" << std::endl;
    std::cout << "d_a is " << d_a << std::endl;
    cudaStatus = GraphPoolAttrGet(0, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GraphPoolAttrGet failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphLaunch(graphExec, stream);//launch graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed!");
        return cudaErrorInvalidValue;
    }

    std::cout << std::endl << "------after launch-------" << std::endl;
    std::cout << "d_a is " << d_a << std::endl;
    cudaStatus = GraphPoolAttrGet(0, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GraphPoolAttrGet failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamSynchronize(stream);//stream synchronization
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed!");
        return cudaErrorInvalidValue;
    }


    std::cout << std::endl << "------after synchronize-------" << std::endl;
    std::cout << "d_a is " << d_a << std::endl;
    cudaStatus = GraphPoolAttrGet(0, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GraphPoolAttrGet failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaDeviceGraphMemTrim(0);//TrimTo
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGraphMemTrim failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << std::endl << "------after trimto-------" << std::endl;
    std::cout << "d_a is " << d_a << std::endl;
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
    cudaStatus = test();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test failed!");
        return 1;
    }
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

