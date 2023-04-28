/*This code tests:An executable graph is launched into multiple streams, even if there is no malloc and free nodes, it can only be serialized;
a graph which has no alloc nodes can be instantiated to multiple executable graphs; 
a graph which has alloc nodes cannot be instantiated to multiple executable graphs.
*/
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define LOOPTIMES 10
#define SIZE 32*1024*1024

__global__ void clockBlock(clock_t clock_count) {
    unsigned int start_clock = (unsigned int)clock();

    clock_t clock_offset = 0;

    while (clock_offset < clock_count) {
        unsigned int end_clock = (unsigned int)clock();
        clock_offset = (clock_t)(end_clock - start_clock);
    }
}
void prepareAllocParams(cudaMemAllocNodeParams* allocParams, size_t bytes,
    int device) {
    memset(allocParams, 0, sizeof(*allocParams));

    allocParams->bytesize = bytes;
    allocParams->poolProps.allocType = cudaMemAllocationTypePinned;
    allocParams->poolProps.location.id = device;
    allocParams->poolProps.location.type = cudaMemLocationTypeDevice;
}

cudaError_t createGraphWithMalloc(cudaGraphExec_t* graphExec) {
    cudaError_t cudaStatus;
    cudaGraph_t graph;
    cudaStatus = cudaGraphCreate(&graph, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream;
    int* d_a = NULL;
    float kernelTime = 5000;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);
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
    clockBlock <<<1, 1, 0, stream >>> (time_clocks);
    cudaStatus = cudaFreeAsync(d_a,stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFreeAsync failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphDestroy(graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t createGraphWithoutMalloc(cudaGraphExec_t* graphExec) {
    cudaError_t cudaStatus;
    cudaGraph_t graph;
    cudaStatus = cudaGraphCreate(&graph, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream;
    float kernelTime = 5000;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);
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

    clockBlock << <1, 1, 0, stream >> > (time_clocks);

    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphDestroy(graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t test1() {
    cudaError_t cudaStatus;
    cudaGraphExec_t graphExec;
    cudaStream_t stream[LOOPTIMES];    
    cudaStatus = createGraphWithMalloc(&graphExec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphDestroy failed!");
            return cudaErrorInvalidValue;
        }
    }
        
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaGraphLaunch(graphExec, stream[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphDestroy failed!");
            return cudaErrorInvalidValue;
        }
    }
    return cudaStatus;
}
cudaError_t test2() {
    cudaError_t cudaStatus;
    cudaGraphExec_t graphExec;
    cudaStream_t stream[LOOPTIMES];
    cudaStatus = createGraphWithoutMalloc(&graphExec);
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
        if (cudaStatus != cudaSuccess) {
             fprintf(stderr, "cudaGraphDestroy failed!");
             return cudaErrorInvalidValue;
        }   
    }
       
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaGraphLaunch(graphExec, stream[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphDestroy failed!");
            return cudaErrorInvalidValue;
        }
    }
    return cudaStatus;
}

cudaError_t test3() {
    cudaError_t cudaStatus;
    cudaGraphExec_t graphExec[LOOPTIMES];
    cudaStream_t stream[LOOPTIMES];
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaStreamCreateWithFlags failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = createGraphWithMalloc(&(graphExec[i]));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "createGraphWithMalloc failed!");
            return cudaErrorInvalidValue;
        }
    }

    for (int i = 0; i < LOOPTIMES; i++){
        cudaStatus = cudaGraphLaunch(graphExec[i], stream[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphLaunch failed!");
            return cudaErrorInvalidValue;
        }
    }
    return cudaStatus;
}

cudaError_t test4() {
    cudaError_t cudaStatus;
    cudaGraphExec_t graphExec[LOOPTIMES];
    cudaStream_t stream[LOOPTIMES];
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphLaunch failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = createGraphWithoutMalloc(&(graphExec[i]));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphLaunch failed!");
            return cudaErrorInvalidValue;
        }
    }

    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaGraphLaunch(graphExec[i], stream[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphLaunch failed!");
            return cudaErrorInvalidValue;
        }
    }
    return cudaStatus;
}

cudaError_t test5() {
    cudaError_t cudaStatus;
    cudaGraph_t graph;
    cudaStatus = cudaGraphCreate(&graph, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaGraphExec_t graphExec1, graphExec2;
    cudaStream_t stream;
    float* d_a = NULL;
    cudaMemAllocNodeParams allocParamsA;
    cudaGraphNode_t allocNodeA,  freeNodeA;
    prepareAllocParams(&allocParamsA, 32 * 1024 * 1024, 0);
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphAddMemAllocNode(&allocNodeA, graph, NULL, 0, &allocParamsA);//create graph
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

    cudaStatus = cudaGraphInstantiate(&graphExec1, graph, NULL, NULL, 0);//instantiate graph first time
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "first time cudaGraphInstantiate failed!");
        exit(0);
    }
    cudaStatus = cudaGraphInstantiate(&graphExec2, graph, NULL, NULL, 0);//instantiate graph second time
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "second time cudaGraphInstantiate failed!");
        exit(0);
    }
    return cudaStatus;

}

cudaError_t test6() {
    cudaError_t cudaStatus;
    cudaGraphExec_t graphExec1, graphExec2;
    cudaStream_t stream,stream1,stream2;
    cudaGraph_t graph;
    cudaStatus = cudaGraphCreate(&graph, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphCreate failed!");
        return cudaErrorInvalidValue;
    }
    float kernelTime = 5000;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);
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
    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed!");
        return cudaErrorInvalidValue;
    }
    clockBlock <<<1, 1, 0, stream >>> (time_clocks);
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }


    cudaStatus = cudaGraphInstantiate(&graphExec1, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "1 cudaGraphInstantiate failed!");
        exit(0);
    }
    cudaStatus = cudaGraphInstantiate(&graphExec2, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "2 cudaGraphInstantiate failed!");
        exit(0);
    }

    cudaStatus = cudaGraphLaunch(graphExec1, stream1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphLaunch(graphExec2, stream2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed!");
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
    //Viewing results in visual profiler
    cudaStatus =test1();//Single executable graph is launched into multiple streams Has malloc nodes
    //cudaStatus =test2();//Single executable graph is launched into multiple streams No malloc nodes
    //cudaStatus =test3();//Multiple executable graphs are launched into multiple streams Have malloc nodes
    //cudaStatus =test4();//Multiple executable graphs are launched into multiple streams No malloc nodes
    //cudaStatus = test5();//A graph with memory allocation nodes cannot instantiate multiple executable graphs
    //cudaStatus =test6();//A graph without memory allocation nodes can instantiate multiple executable graphs
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

