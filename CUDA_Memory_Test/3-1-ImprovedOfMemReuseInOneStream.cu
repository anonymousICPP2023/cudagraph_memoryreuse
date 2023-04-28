/*This code tests:The effect of performance improvement of in-stream reuse*/
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>


#define LOOPTIMES 20  //loop times
#define SIZE 64*1024*1024  //size of allocation



__global__ void clockBlock(clock_t clock_count) { //kernel
    unsigned int start_clock = (unsigned int)clock();

    clock_t clock_offset = 0;

    while (clock_offset < clock_count) {
        unsigned int end_clock = (unsigned int)clock();
        clock_offset = (clock_t)(end_clock - start_clock);
    }
}

cudaError_t test1() {
    std::cout << std::endl << "This code tests: The effect of performance improvement of in-stream reuse" << std::endl << std::endl;
    std::cout << "cudaMallocAsync+cudaFreeAsync---No synchronize in each loop" << std::endl << std::endl;
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

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);



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
        clockBlock <<<1, 1, 0, stream >>> (time_clocks);
        cudaStatus = cudaFreeAsync(d_a, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaFreeAsync failed!");
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
    std::cout << std::endl << "This code tests: The effect of performance improvement of in-stream reuse" << std::endl << std::endl;
    std::cout << "cudaMallocAsync+cudaFreeAsync---Synchronize in each loop" << std::endl << std::endl;
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

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);



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
        clockBlock << <1, 1, 0, stream >> > (time_clocks);
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
cudaError_t test3() {
    std::cout << std::endl << "This code tests: The effect of performance improvement of in-stream reuse" << std::endl << std::endl;
    std::cout << "cudaMallocAsync---Memory is only allocated but not released" << std::endl << std::endl;
    cudaError_t cudaStatus;
    int* device[LOOPTIMES];
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

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);



    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaMallocAsync((void**)&device[i], SIZE, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocAsync failed!");
            return cudaErrorInvalidValue;
        }
        clockBlock <<<1, 1, 0, stream >>> (time_clocks);
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

    printf("test3 time is %f\n", time);

    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaFreeAsync(device[i], stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaFreeAsync failed!");
            return cudaErrorInvalidValue;
        }
    }

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
cudaError_t test4() {
    std::cout << std::endl << "This code tests: The effect of performance improvement of in-stream reuse" << std::endl << std::endl;
    std::cout << "cudaMallocAsync+cudaFreeAsync---Set threshold max and synchronize" << std::endl << std::endl;
    cudaError_t cudaStatus;
    int* d_a = NULL;
    cudaStream_t stream;
    cudaMemPool_t memPool;
    cudaStatus = cudaDeviceGetDefaultMemPool(&memPool, 0); 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetDefaultMemPool failed!");
        return cudaErrorInvalidValue;
    }

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

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);
    long unsigned int setVal = UINT64_MAX;
    cudaStatus = cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolSetAttribute failed!");
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
        clockBlock << <1, 1, 0, stream >> > (time_clocks);
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
    printf("test4 time is %f\n", time);
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
cudaError_t test5() {
    std::cout << std::endl << "This code tests: The effect of performance improvement of in-stream reuse" << std::endl << std::endl;
    std::cout << "cudaMallocAsync+cudaFreeAsync---Set threshold 32MB and synchronize" << std::endl << std::endl;
    cudaError_t cudaStatus;
    int* d_a = NULL;
    cudaStream_t stream;
    cudaMemPool_t memPool;
    cudaStatus = cudaDeviceGetDefaultMemPool(&memPool, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetDefaultMemPool failed!");
        return cudaErrorInvalidValue;
    }

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

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);
    long unsigned int setVal = 32*1024*1024;
    cudaStatus = cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolSetAttribute failed!");
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
        clockBlock << <1, 1, 0, stream >> > (time_clocks);
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
    printf("test5 time is %f\n", time);
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
cudaError_t test6() {
    std::cout << std::endl << "This code tests: The effect of performance improvement of in-stream reuse" << std::endl << std::endl;
    std::cout << "cudaMalloc+cudaFree" << std::endl << std::endl;
    cudaError_t cudaStatus;
    int* d_a = NULL;
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

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);



    cudaStatus = cudaEventRecord(start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaMalloc((void**)&d_a, SIZE);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocAsync failed!");
            return cudaErrorInvalidValue;
        }
        clockBlock << <1, 1, 0 >> > (time_clocks);
        cudaStatus = cudaFree(d_a);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaFreeAsync failed!");
            return cudaErrorInvalidValue;
        }
    }

    cudaStatus = cudaEventRecord(stop);
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
    printf("test6 time is %f\n", time);
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
    return cudaSuccess;
}
cudaError_t test7() {
    std::cout << std::endl << "This code tests: The effect of performance improvement of in-stream reuse" << std::endl << std::endl;
    std::cout << "cudaMalloc---Memory is only allocated and not freed" << std::endl << std::endl;
    cudaError_t cudaStatus;
    int* device[LOOPTIMES];
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

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);

    cudaStatus = cudaEventRecord(start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaMalloc((void**)&device[i], SIZE);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocAsync failed!");
            return cudaErrorInvalidValue;
        }
        clockBlock << <1, 1, 0 >> > (time_clocks);
    }

    cudaStatus = cudaEventRecord(stop);
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
    printf("test7 time is %f\n", time);

    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaFree(device[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaFreeAsync failed!");
            return cudaErrorInvalidValue;
        }
    }
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

  //  cudaStatus = test1();//cudaMallocAsync+cudaFreeAsync---Do not synchronize in each loop---Memory is reused
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test1 failed!");
        return 1;
    } 
  //  cudaStatus = test2();//cudaMallocAsync+cudaFreeAsync---Synchronize in each loop---Memory is not reused
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test2 failed!");
        return 1;
    }
   // cudaStatus = test3();//cudaMallocAsync---Memory is only allocated but not released---Memory is not reused
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test3 failed!");
        return 1;
    }
    //cudaStatus = test4();//cudaMallocAsync+cudaFreeAsync---Set threshold max and synchronize---Memory is reused
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test4 failed!");
        return 1;
    }
   // cudaStatus = test5();//cudaMallocAsync+cudaFreeAsyncc---Set threshold 32MB and synchronize---Partial memory is reused
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test5 failed!");
        return 1;
    }
    //cudaStatus = test6();//cudaMalloc+cudaFree
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test6 failed!");
        return 1;
    }
    cudaStatus = test7();//cudaMalloc---Memory is only allocated and not freed
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test7 failed!");
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

