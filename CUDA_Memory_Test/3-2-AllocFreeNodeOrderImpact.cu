/*This code tests: compare allocA->freeA->allocB with allocA->allocB->freeA*/
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>


#define SIZE 416*1024*1024


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


cudaError_t test1() {
    cudaError_t cudaStatus;
    int* d_a = NULL,*d_b = NULL;
    cudaStream_t stream;
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
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
    cudaGraph_t graph;
    cudaStatus = cudaGraphCreate(&graph, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaGraphExec_t graphExec;
    struct usageStatistics statistics = { 0,0,0,0 };


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

    cudaStatus = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaGraphLaunch(graphExec, stream);//launch graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed!");
        return cudaErrorInvalidValue;
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
    printf("allocA->freeA->allocB is %f\n", time);
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
cudaError_t test2() {
    cudaError_t cudaStatus;
    int* d_a = NULL, * d_b = NULL;
    cudaStream_t stream;
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
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
    cudaGraph_t graph;
    cudaStatus = cudaGraphCreate(&graph, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaGraphExec_t graphExec;
    struct usageStatistics statistics = { 0,0,0,0 };


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
    cudaStatus = cudaMallocAsync((void**)&d_b, SIZE, stream);
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

    cudaStatus = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaGraphLaunch(graphExec, stream);//launch graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed!");
        return cudaErrorInvalidValue;
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
    printf("allocA->allocB->freeA is %f\n", time);
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

    std::cout << std::endl << "This code tests:" << " compare allocA->freeA->allocB with allocA->allocB->freeA" << std::endl << std::endl;
    cudaStatus = test1();//allocA->freeA->allocB
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test1 failed!");
        return 1;
    }
    cudaStatus = test2();//allocA->allocB->freeA
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test2 failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;

}

