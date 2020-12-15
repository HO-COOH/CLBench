# CLBench
This is an OpenCL benchmark that examine an end-to-end performance of a typical OpenCL application, which is part of my master degree project.

## What is benchmarked?
The benchmark will run the following testing on **your default GPU**. On a laptop, this is usually your integrated GPU.
- Data transfer
  + host -> device
  + device -> host
- Kernel compilation
  + compile from source string (both single-threaded & multi-threaded)
  + compile from saved binary (both single-threaded & multi-threaded)
- Some mathematical operations
  - Reduction
  - Matrix multiplication
  - Convolution
Note: Some of the benchmark may fail on your GPU. Do NOT use the kernels in the project for real-world application, they are only naive implementations.
## Dependency
I packaged dependencies (dll and lib) in [./dependency](./dependency) for Windows 10 64bit, so it should build and run without any additional step.

Otherwise, use [vcpkg](https://github.com/microsoft/vcpkg) to install `OpenCL` with the command:
```
vcpkg install OpenCL
```
Then do your usual `CMAKE_TOOLCHAIN_FILE` stuff which I do not bother to write here :)

## Sample output
Below is an example of running the project on my 1660 Super
```
///////////Host -> Device///////////
Testing <CL_MEM_COPY_HOST_PTR> 0.00390625 MB -> 312.5 MB/s
Testing <CL_MEM_COPY_HOST_PTR> 1 MB -> 4089.98 MB/s
Testing <CL_MEM_COPY_HOST_PTR> 32 MB -> 4744.33 MB/s
Testing <CL_MEM_COPY_HOST_PTR> 512 MB -> 5391.13 MB/s
Testing <CL_MEM_COPY_HOST_PTR> 1024 MB -> 5265.75 MB/s
Testing <CL_MEM_COPY_HOST_PTR> 2048 MB -> 5275.61 MB/s
Testing <CL_MEM_COPY_HOST_PTR> 4096 MB -> 5367.32 MB/s
Testing <CL_MEM_COPY_HOST_PTR> 6144 MB -> 5288.88 MB/s
Testing <clEnqueueWriteBuffer> 0.00390625 MB -> 5.66698 MB/s
Testing <clEnqueueWriteBuffer> 1 MB -> 2070.82 MB/s
Testing <clEnqueueWriteBuffer> 32 MB -> 4873.29 MB/s
Testing <clEnqueueWriteBuffer> 512 MB -> 5614.5 MB/s
Testing <clEnqueueWriteBuffer> 1024 MB -> 5644.04 MB/s
Testing <clEnqueueWriteBuffer> 2048 MB -> 5539.07 MB/s
Testing <clEnqueueWriteBuffer> 4096 MB -> 5788.36 MB/s
Testing <clEnqueueWriteBuffer> 6144 MB ->
Testing <clEnqueueWriteBuffer> failed: Code: -4 clEnqueueWriteBuffer
Testing <clEnqueueMapBuffer> 0.00390625 MB -> 2.86005 MB/s
Testing <clEnqueueMapBuffer> 1 MB -> 768.226 MB/s
Testing <clEnqueueMapBuffer> 32 MB -> 1819.25 MB/s
Testing <clEnqueueMapBuffer> 512 MB -> 2161.07 MB/s
Testing <clEnqueueMapBuffer> 1024 MB -> 2397.11 MB/s
Testing <clEnqueueMapBuffer> 2048 MB -> 2462.09 MB/s
Testing <allocate + clEnqueueWriteBuffer> 0.00390625 MB -> 5.85118 MB/s
Testing <allocate + clEnqueueWriteBuffer> 1 MB -> 1533.74 MB/s
Testing <allocate + clEnqueueWriteBuffer> 32 MB -> 4796.31 MB/s
Testing <allocate + clEnqueueWriteBuffer> 512 MB -> 5353.68 MB/s
Testing <allocate + clEnqueueWriteBuffer> 1024 MB -> 4712.38 MB/s
Testing <allocate + clEnqueueWriteBuffer> 2048 MB -> 5553.98 MB/s
Testing <allocate + clEnqueueWriteBuffer> 4096 MB -> 5474.66 MB/s
Testing <allocate + clEnqueueWriteBuffer> 6144 MB ->
Testing <clEnqueueWriteBuffer> failed: Code: -4 clEnqueueWriteBuffer
Testing <allocate + clEnqueueMapBuffer> 0.00390625 MB -> 2.27081 MB/s
Testing <allocate + clEnqueueMapBuffer> 1 MB -> 554.508 MB/s
Testing <allocate + clEnqueueMapBuffer> 32 MB -> 1493.6 MB/s
Testing <allocate + clEnqueueMapBuffer> 512 MB -> 1523.53 MB/s
Testing <allocate + clEnqueueMapBuffer> 1024 MB -> 1669.59 MB/s
Testing <allocate + clEnqueueMapBuffer> 2048 MB -> 1749.95 MB/s


////////////Device -> Host///////////
Testing <clEnqueueReadBuffer> 0.00390625 MB -> 65.8727 MB/s
Testing <clEnqueueReadBuffer> 1 MB -> 3411.8 MB/s
Testing <clEnqueueReadBuffer> 32 MB -> 6107.8 MB/s
Testing <clEnqueueReadBuffer> 512 MB -> 6304.11 MB/s
Testing <clEnqueueReadBuffer> 1024 MB -> 6257.96 MB/s
Testing <clEnqueueReadBuffer> 2048 MB -> 6023.15 MB/s
Testing <clEnqueueMapBuffer> 0.00390625 MB -> 5.17247 MB/s
Testing <clEnqueueMapBuffer> 1 MB -> 1181.75 MB/s
Testing <clEnqueueMapBuffer> 32 MB -> 2465.22 MB/s
Testing <clEnqueueMapBuffer> 512 MB -> 2416.68 MB/s
Testing <clEnqueueMapBuffer> 1024 MB -> 2369.02 MB/s
Testing <clEnqueueMapBuffer> 2048 MB -> 2476.68 MB/s
Testing <CompileSingleThread> -> 1239.43 kernels /s
14037 microsec
Testing <CompilingMultiThreadAsync> -> 1528.34 kernels /s
11465 microsec
Testing <CompilingMultiThread> -> 1439.45 kernels /s
12092 microsec
Testing <CompileSingleThread> -> 424.445 kernels /s
40500 microsec
Testing <LoadingBinarySingleThread> -> Loaded 17 kernels from binary 2606.28 kernels /s
6736 microsec
Testing <CompileSingleThread> -> 415.394 kernels /s
41397 microsec
Testing <LoadingBinaryMultiThread> -> Loaded 17 kernels from binary 2571.43 kernels /s
6809 microsec
Testing <ReduceInterleaved> with 262144
1.06997 GB/s Round = 3
1253 microsec
Reduce result: -338.325
Testing <ReduceInterleaved> with 524288
1.29406 GB/s Round = 3
1862 microsec
Reduce result: 225.024
Testing <ReduceInterleaved> with 1048576
1.84509 GB/s Round = 3
2607 microsec
Reduce result: -182.728
Testing <ReduceInterleaved> with 2097152
2.55879 GB/s Round = 3
3549 microsec
Reduce result: 120.716
Testing <ReduceInterleaved> with 4194304
3.08496 GB/s Round = 3
5531 microsec
Reduce result: -489.62
Testing <ReduceInterleaved> with 8388608
3.53403 GB/s Round = 3
9343 microsec
Reduce result: 400.092
Testing <ReduceInterleaved> with 16777216
3.78492 GB/s Round = 4
17107 microsec
Reduce result: -2265.84
Testing <ReduceInterleaved> with 33554432
3.95854 GB/s Round = 4
32094 microsec
Reduce result: 2501.33
Testing <ReduceInterleaved> with 67108864
4.09666 GB/s Round = 4
61556 microsec
Reduce result: -1839.39
Testing <ReduceInterleavedNonDivergent> with 262144
1.14944 GB/s Round = 3
1205 microsec
Reduce result: -664.753
Testing <ReduceInterleavedNonDivergent> with 524288
1.22017 GB/s Round = 3
2113 microsec
Reduce result: 391.283
Testing <ReduceInterleavedNonDivergent> with 1048576
1.93724 GB/s Round = 3
2427 microsec
Reduce result: 583.619
Testing <ReduceInterleavedNonDivergent> with 2097152
2.49314 GB/s Round = 3
3775 microsec
Reduce result: -51.6634
Testing <ReduceInterleavedNonDivergent> with 4194304
3.12063 GB/s Round = 3
5484 microsec
Reduce result: 51.8917
Testing <ReduceInterleavedNonDivergent> with 8388608
3.53651 GB/s Round = 3
9346 microsec
Reduce result: -70.9239
Testing <ReduceInterleavedNonDivergent> with 16777216
3.76654 GB/s Round = 4
17086 microsec
Reduce result: 51.5029
Testing <ReduceInterleavedNonDivergent> with 33554432
4.03851 GB/s Round = 4
31471 microsec
Reduce result: -2708.48
Testing <ReduceInterleavedNonDivergent> with 67108864
4.17352 GB/s Round = 4
60432 microsec
Reduce result: -3104.14
Testing <ReduceSequential> with 262144
1.21102 GB/s Round = 3
1105 microsec
Reduce result: -334.107
Testing <ReduceSequential> with 524288
1.39899 GB/s Round = 3
1711 microsec
Reduce result: -471.559
Testing <ReduceSequential> with 1048576
1.8424 GB/s Round = 3
2599 microsec
Reduce result: -48.9131
Testing <ReduceSequential> with 2097152
2.3791 GB/s Round = 3
3797 microsec
Reduce result: 291.875
Testing <ReduceSequential> with 4194304
3.09388 GB/s Round = 3
5592 microsec
Reduce result: 318.68
Testing <ReduceSequential> with 8388608
3.55461 GB/s Round = 3
9293 microsec
Reduce result: 116.345
Testing <ReduceSequential> with 16777216
3.72765 GB/s Round = 4
17272 microsec
Reduce result: 510.172
Testing <ReduceSequential> with 33554432
4.06113 GB/s Round = 4
31294 microsec
Reduce result: -6119.86
Testing <ReduceSequential> with 67108864
4.21989 GB/s Round = 4
59780 microsec
Reduce result: -11.2635
Testing <FirstAddDuringLoad> with 262144
665 microsec
Testing <NaiveConv> with 4096 x 4096channel = 1 with filter = 3
12.0911 GFlops
Testing <NaiveConv> with 4096 x 4096channel = 1 with filter = 5
26.9653 GFlops
Testing <NaiveConv> with 4096 x 4096channel = 1 with filter = 7
33.088 GFlops
Testing <NaiveConv> with 4096 x 4096channel = 1 with filter = 9
35.1942 GFlops
```

## Further development
- Add GNUPlot for sexy output
- Enable Multi-GPU testing
