/*****************************************************************//**
 * \file   Test.hpp
 * \brief  All tests for the project
 * 
 * \author Peter
 * \date   October 2020
 *********************************************************************/
#pragma once
#include <cstddef>

namespace test
{
    namespace SanityCheck
    {
        /**
         * @brief Test passing a struct to kernel
         */
        void PassingStruct();

        void Transpose();
    }

    namespace DataTransfer
    {
        /**
         * @brief Test the performance of copying data when creating buffer with CL_MEM_COPY_HOST_PTR
         * @param bytes Size for the test data to be copied
         */
        void CopyHostPtr(size_t bytes);

        /**
         * @brief Test the performance of copying data using clEnqueueWriteBuffer
         * @param bytes Size for the test data to be copied
         */
        void WriteBuffer(size_t bytes);

        /**
         * @brief Test the performance of copying data using clEnqueueWriteBuffer plus the time to allocate the buffer
         * @param bytes Size for the test data to be copied
         */
        void WriteBufferTotal(size_t bytes);

        /**
         * @brief Test the performance of copying data using clEnqueueMapBuffer
         * @param bytes Size for the test data to be copied
         */
        void WriteMapBuffer(size_t bytes);

        /**
         * @brief Test the performance of copying data using clEnqueueMapBuffer plus the time to allocate the buffer
         * @param bytes Size for the test data to be copied
         */
        void WriteMapBufferTotal(size_t bytes);

        /**
         * @brief Test the performance of copying data from host -> device
         */
        void CopyToDevice();
        /**
         * @brief Test the performance of reading gpu data using clEnqueueReadBuffer
         * @param bytes Size for the test data to be copied
         */
        void ReadBuffer(size_t bytes);

        /**
         * @brief Test the performance of reading gpu data using clEnqueueMapBuffer
         * @param bytes Size for the test data to be copied
         */
        void ReadMapBuffer(size_t bytes);

        /**
         * @brief Test the performance of copying data from device -> host
         */
        void CopyToHost();

        /**
         * @brief Mapping buffer with CL_MAP_WRITE
         */
        void MapReadWrite();

        /**
         * @brief Test the system RAM I/O speed
         */
        void RamSpeed();

        /**
         * @brief Run all test
         */
        void DataTransfer();
    }

    namespace MemoryType
    {
        /**
         * @brief Allocate buffer with CL_USE_HOST_PTR
         */
        void UseHostPtr();

        /**
         * @brief Allocate buffer with CL_ALLOC_HOST_PTR
         */
        void AllocHostPtr();

        /**
         * @brief Allocate buffer with CL_ALLOC_HOST_PTR|CL_USE_HOST_PTR
         */
        void AllocHostPtrWithUseHostPtr();

    }

    namespace Compilation
    {
        /**
         * @brief Test the performance of compiling OpenCL kernels using 1 thread
         */
        void SingleThread(bool saveKernel = false);

        /**
         * @brief Test the performance of compiling OpenCL kernels using multi-threading with std::async
         */
        void MultiThreadWithAsync();

        /**
         * @brief Test the performance of compiling OpenCL kernels using multi-threading with std::thread
         */
        void MultiThreadWithThread();

        /**
         * @brief Test the performance of loading compiled binary OpenCL kernels
         */
        void LoadFromBinary();

        /**
         * @brief Test the performance of loading compiled binary OpenCL kernels using async
         */
        void MultiThreadLoadFromBinary();

        /**
         * @brief Run all test
         */
        void Compilation();
    }

    namespace Timing
    {
        /**
         * @brief Test the method of using OpenCL profile events versus scoped timer in kernel execution
         */
        void ProfileKernelExecution();

        /**
         * @brief Test the method of using OpenCL profile events versus scoped timer in data transfer
         */
        void ProfileDataTransfer();

    }

    namespace Benchmark
    {
        namespace Reduction
        {
            /**
             * @brief Using std::accumulate
             */
            void StdAccumulate(size_t numElements);

            /**
             * @brief Using std::reduce with default execution policy
             */
            void StdReduce(size_t numElements);

            /**
             * @brief Use interleaved addressing
             * @details Each work-item grab an element indexed at its own id and another element next to it (the interval is multiplied each round)
             */
            void InterleavedAddressingDivergent(size_t numElements);

            /**
             * @brief Use interleaved addressing
             * @details Each work-item grab an element indexed at its own id and another element next to it (the interval is multiplied each round)
             */
            void InterleavedAddressingNonDivergent(size_t numElements);

            /**
             * @brief Use sequential addressing
             * @details Each work-item grab an element indexed at its own id and another element at another half of the array
             */
            void SequentialAddressing(size_t numElements);

            /**
             * @brief Do the first round of reduction when reading the elements
             */
            void FirstAddDuringLoad(size_t numElements);

            /**
             * @brief Test different methods of reduction algorithm
             */
            void Reduction();
        }

        namespace MatrixMultiplication
        {

            /**
             * @brief The naive matrix multiplication implementation
             */
            void Naive(size_t size);

            /**
             * @brief Transpose the right hand side matrix by CPU and do the matrix multiplication on GPU
             */
            void TransposeByCPU(size_t size);

            /**
             * @brief Directly copy the right hand side matrix -> GPU and do the transpose, then do the matrix multiplication
             */
            void TransposeByGPU(size_t size);

            /**
             * @brief Use tiled matrix block multiplication to make use of local memory
             */
            void UseLocalMemory(size_t size);


            /**
             * @brief Use tiled matrix block multiplication to make use of local memory and unrolled the blocked multiplication loop
             */
            void UnrolledMul(size_t size);

            /**
             * @brief Use row-block-row-major ordering
             */
            void RowBlockRowMajorOrdering(size_t size);

            /**
             * @brief Use constant global memory
             */
            void UseConstantMemory(size_t size);

            /**
             * @brief Test different methods of matrix multiplication
             */
            void MatrixMultiplication();

            /**
             * @brief Matrix multiplication by CPU
             */
            void MatrixMultiplicationCPU();
        }

        namespace Convolution
        {
            /* Terminology and assumptions:
             * W: width of input image
             * H: height of input image
             *
             * FS(Filter size): The edge size of a square matrix, usually an odd number like 3, 5...
             * FA(Filter area): FS*FS
             * HFS(Half filter size): floor(FS/2), and because FS is an odd number, HFS is also odd
             * */

            /**
             * @brief Use the naive implementation of convolution
             * @details
             * Launch W*H work items.
             * Each work item grab FA pixels from the input image and a copy of filter.
             */
            void Naive(size_t pixel);

            /**
             * @brief Use vectorized data type
             */
            void Vector(size_t pixel);

            /**
             * @brief Use loop unrolling
             */
            void LoopUnroll(size_t pixel);

            /**
             * @brief Make use of local memory and work-group
             * @details In Naive implementation, each work-item need to read FA neighboring pixels around the center from the global memory,
             * therefore its neighboring work-items are reading overlapping global memory regions, which can be eliminated by making use of local-memory
             *
             */
            void GroupedConv(size_t pixel);

            /**
             * @brief Test different methods of convolution
             */
            void Convolution();
        }
    }
}
