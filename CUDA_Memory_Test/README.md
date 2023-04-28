# Memory reuse in CUDA Graph

These test cases are used to clarify the details in programming memory reuse in CUDA Graph. According to our programming experience,  it is challenging for programmers to reuse memory in CUDA graph **correctly** and **efficiently**. 

## (1) test cases for Section 3.1 in ICPP paper

### 3-1-ImprovedOfMemReuseInOneStream.cu 

A sample shows that  the effect of improved performance of memory reuse in one stream:  

  - test1() use cudaMallocAsync with cudaFreeAsync, do not synchronize the stream in each loop, the result shows that the memory is reused.  
  - test2() use cudaMallocAsync with cudaFreeAsync, synchronize the stream in each loop, the result shows that the memory is not reused. 
  - test3() only use cudaMallocAsync, allocate memory without freeing memory, the result shows that the memory is not reused.  
  - test4() use cudaMallocAsync with cudaFreeAsync, set the threshold to UNIT_MAX and synchronize the stream in each loop, the result shows that the memory is reused.  
  - test5() use cudaMallocAsync with cudaFreeAsync, set the threshold to 32MB and synchronize the stream in each loop, the result shows that part of the memory is reused.  
  - test6() use cudaMalloc with cudaFree.  
  - test7() only use cudaMalloc, allocate memory without freeing memory.  

### 3-1-MemPoolGranularity.cu 

A sample confirms that  the memory pool has a granularity of 32MB.

### 3-1-FragmentsOfMemPool.cu

A sample shows that the memory pool will not be defragmented during the stream operations.  And, the memory pool will be managed with a simple strategy and the fragments will be handled at a subsequent time.

## (2) test cases for Section 3.2  in ICPP paper

### 3-2-AllocFreeNodeOrderImpact.cu

A sample compares that with the differences between two kinds of execution orders: allocA->freeA->allocB and allocA->allocB->freeA.

## (3) test cases for Section 3.3  in ICPP paper

### 3-3-MemFootprintOfDifferentNewAllocSize.cu

A sample tests that the situation of memory reuse: the memory size requested by new alloc node is larger than, less than or equal to the memory size requested by alloc node.

### 3-3-2NodesReuse1Node.cu

A sample tests that whether or not can the memory of the previous node be reused when memory size of the previous node is 1GB, and memory size of the two subsequent nodes is 0.5GB.

## (4) test cases for Section 3.4  in ICPP paper

### 3-4-GraphUnusedMemNotReleaseAfterSync.cu

A sample shows that all unused memory in the graph memory pool is not released back to the OS during every synchronization operation.

### 3-4-HowMemPoolThresholdWork.cu

A sample  shows that two conclusions: if the threshold is set, the physical memory will be released to the threshold during every synchronization operation; if the threshold is not set, the physical memory will be fully released during every synchronization operation.

### 3-4-PerformanceOfSetThreshold.cu

A sample compares that performance differences of cudaMallocAsync between setting the threshold and do not set the threshold. 

### 3-4-SetThreshodNotAffectPhysicalMem.cu 

A sample shows that the total physical memory does not change after setting the threshold for the memory pool.

### 3-4-GraphMemPoolIsDifferentFromSOMA.cu

A sample confirms that the allocation comes from the graph memory pool instead of the default memory pool when launching an allocation graph.

### 3-4-DifferentGraphShareOneMemPool.cu 

A sample that shows all graphs share the same memory pool.

## (5) test cases for Section 3.5  in ICPP paper

### 3-5-GraphMemReuse.cu

A sample shows three finds: (1) An instantiated graph which is launched in multiple streams, even if there is no alloc and free nodes in this graph, it can only be executed serially; (2) A graph which has no alloc nodes can be instantiated to multiple executable graphs; (3) A graph which has alloc nodes cannot be instantiated to multiple executable graphs.


## (6) test cases for Section 4.1  in ICPP paper

### 4-1-GraphMemShouldNotFrequentTrim.cu

A sample shows that graph memory should avoid frequent trim operations.

### 4-1-GraphUploadReduceLaunchOverhead.cu

A sample shows that cudaGraphUpload can complete the mapping first and move it out of critical path, which can significantly reduce the launch overhead.


## (7) Evaluation Platform

Our study have been conducted on a NVIDIA GPU V100 server.  However,  other GPUs such as A100 and H100 should also be supported.  All kinds of Linux such as Ubuntu 20.04.

## (8) Prerequisites

GPU driver should be installed, then download and install the [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-downloads) or above.  Our test cases are only evaluated in CUDA version 11.7.

## (9) Build

The Linux samples are built using Makefile. To use the Makefile, change the current directory to the sample directory you wish to build, and run make:

```
$cd <sample_dir>
$make
```

If you want to remove all .o files

```
$cd <sample_dir>
$make clean
```
## (10) Run Test Cases

```
$cd <sample_dir>
$./test_case
```