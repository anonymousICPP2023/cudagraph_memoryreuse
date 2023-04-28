/*This code tests: All graphs share one memory pool*/

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>


#define SIZE 32*1024*1024

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
    int device = 0;
    cudaError_t cudaStatus;
    cudaGraphExec_t graphExec1,graphExec2,graphExec3;
    cudaGraph_t graph;  
    struct usageStatistics u = { 0,0,0,0 };
    cudaStatus = cudaGraphCreate(&graph, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream, stream1,stream2, stream3;
    int* d_a = NULL;
    int* d_b = NULL;
    int* d_c = NULL;

    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
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
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphInstantiate(&graphExec1, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMallocAsync((void**)&d_b, SIZE, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocAsync failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphInstantiate(&graphExec2, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMallocAsync((void**)&d_c, SIZE, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocAsync failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphInstantiate(&graphExec3, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }

    std::cout << std::endl << "------before launch-------" << std::endl;
    std::cout << "d_a is " << d_a << std::endl;
    std::cout << "d_b is " << d_b << std::endl;
    std::cout << "d_c is " << d_c << std::endl;

    cudaStatus = cudaGraphLaunch(graphExec1, stream1);//launch graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = GraphPoolAttrGet(device, &u);//Query graph memory pool usage 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GraphPoolAttrGet failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphLaunch(graphExec2, stream2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = GraphPoolAttrGet(device, &u);//Query graph memory pool usage 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GraphPoolAttrGet failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphLaunch(graphExec3, stream3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = GraphPoolAttrGet(device, &u);//Query graph memory pool usage 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GraphPoolAttrGet failed!");
        return cudaErrorInvalidValue;
    }

    std::cout << std::endl<< "------after launch-------" << std::endl;
    std::cout << "d_a is " << d_a << std::endl;
    std::cout << "d_b is " << d_b << std::endl;
    std::cout << "d_c is " << d_c << std::endl;



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

    cudaStatus = test();
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

