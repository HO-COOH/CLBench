#include "Test.hpp"
#include "GPU.h"
#include "Timer.hpp"
#include "SizeLiteral.hpp"
#include <future>
#include <iostream>
#include <numeric>
#include <type_traits>
#include "Error.hpp"
#include <algorithm>

#include "System.h"
#include "Matrix.hpp"
#include "Image.hpp"

#ifdef ANDROID
#include <array>
#endif

#ifndef ANDROID
    #include <execution>
    #include <filesystem>
#endif

template<typename Lhs, typename Rhs>
constexpr std::common_type_t<Lhs, Rhs> abs(Lhs lhs, Rhs rhs)
{
    return lhs >= rhs ? (lhs - rhs) : (rhs - lhs);
}

namespace test
{
    namespace SanityCheck
    {
        void PassingStruct()
        {
            struct Thing
            {
                int data[3];
            };

            Thing thing{ 1,2,3 };
            gpu.enqueueKernel(gpu["PassingStruct"], std::forward_as_tuple(thing), {}, { 1 });
            gpu.finish();
        }

        void Transpose()
        {
            using Benchmark::MatrixMultiplication::Matrix;
            for (size_t size : { 128, 256, 512, 1024 })
            {
                Matrix m = Matrix::make_test_matrix(size, size);
                Matrix n{ size, size, Matrix::NoAlloc{} };

                auto buf = gpu.malloc<float, AccessMode::Read>(m.size(), m.data);
                auto result_buf = gpu.malloc<float, AccessMode::Write>(m.size());

                gpu.enqueueKernel(gpu["Transpose"], std::make_tuple(buf.getClBuffer(), result_buf.getClBuffer()), {}, { size, size });
                auto mappedResult = result_buf.map<AccessMode::Read>();
                n.data = mappedResult.m_ptr;
                std::cout << "Transpose " << size << "finished\n";
            }
            //std::cout << n << '\n';
        }

    }

    static void PrintFailureMessage(const char* msg, cl::Error const& err) noexcept
    {
        std::cerr << '\n' << msg << "Code: " << err.err() << ' ' << err.what() << '\n';
    }

    namespace DataTransfer
    {
        template<typename T>
        static void MakeData(T* ptr, size_t bytes)
        {
            ptr[0] = 1;
            ptr[bytes - 1] = 2;
        }



        //void PrintDeviceAndTest(ComputeDevice const& device, const char* TestName)

        /**
         * @brief Test the performance of copying data when creating buffer with CL_MEM_COPY_HOST_PTR
         * @param bytes Size for the test data to be copied
         */
        void CopyHostPtr(size_t bytes)
        {
            try {

                std::cout << "Testing <CL_MEM_COPY_HOST_PTR> " << toMb(bytes) << " MB -> ";
                auto const ptr = std::make_unique<char[]>(bytes);
                MakeData(ptr.get(), bytes);

                Timer<false> t;
                {
                    auto gpuBuffer = gpu.malloc<char, AccessMode::Read>(bytes, ptr.get());
                    //gpu.enqueueKernel(gpu["test"], std::make_tuple(gpuBuffer.getClBuffer(), bytes), { 0 }, { 1 });
                }
                std::cout << t.perSec(toMb(bytes)) << " MB/s\n";
            }
            catch (cl::Error const& err)
            {
                PrintFailureMessage("Testing <CL_MEM_COPY_HOST_PTR> failed: ", err);
                throw;
            }
        }

        /**
         * @brief Test the performance of copying data using clEnqueueWriteBuffer
         * @param bytes Size for the test data to be copied
         */
        void WriteBuffer(size_t bytes)
        {
            try {
                std::cout << "Testing <clEnqueueWriteBuffer> " << toMb(bytes) << " MB -> ";
                auto gpuBuffer = gpu.malloc<char, AccessMode::Read>(bytes);
                auto const ptr = std::make_unique<char[]>(bytes);
                MakeData(ptr.get(), bytes);

                Timer<false> t;
                {

                    gpuBuffer.copyFrom(ptr.get(), bytes, true);
                    //gpu.enqueueKernel(gpu["test"], std::make_tuple(gpuBuffer.getClBuffer(), bytes), { 0 }, { 1 });
                }
                std::cout << t.perSec(toMb(bytes)) << " MB/s\n";
            }
            catch (cl::Error const& err)
            {
                PrintFailureMessage("Testing <clEnqueueWriteBuffer> failed: ", err);
                throw;
            }
        }

        void WriteBufferTotal(size_t bytes)
        {
            try {
                std::cout << "Testing <allocate + clEnqueueWriteBuffer> " << toMb(bytes) << " MB -> ";
                auto const ptr = std::make_unique<char[]>(bytes);
                MakeData(ptr.get(), bytes);

                Timer<false> t;
                {
                    auto gpuBuffer = gpu.malloc<char, AccessMode::Read>(bytes);
                    gpuBuffer.copyFrom(ptr.get(), bytes, true);
                }
                std::cout << t.perSec(toMb(bytes)) << " MB/s\n";
            }
            catch (cl::Error const& err)
            {
                PrintFailureMessage("Testing <clEnqueueWriteBuffer> failed: ", err);
                throw;
            }
        }

        /**
         * @brief Test the performance of copying data using clEnqueueMapBuffer
         * @param bytes Size for the test data to be copied
         */
        void WriteMapBuffer(size_t bytes)
        {
            try {
                std::cout << "Testing <clEnqueueMapBuffer> " << toMb(bytes) << " MB -> ";
                auto gpuBuffer = gpu.malloc<char, AccessMode::Read>(bytes);
                auto const ptr = std::make_unique<char[]>(bytes);
                MakeData(ptr.get(), bytes);

                Timer<false> t;
                {
                    auto mappedBuffer = gpuBuffer.map<AccessMode::Write>();
                    std::copy_n(ptr.get(), bytes, mappedBuffer.m_ptr);
                }
                std::cout << t.perSec(toMb(bytes)) << " MB/s\n";
            }
            catch (cl::Error const& err)
            {
                PrintFailureMessage("Testing <clEnqueueMapBuffer> failed: ", err);
                throw;
            }
        }

        void WriteMapBufferTotal(size_t bytes)
        {
            try {
                std::cout << "Testing <allocate + clEnqueueMapBuffer> " << toMb(bytes) << " MB -> ";
                
                auto const ptr = std::make_unique<char[]>(bytes);
                MakeData(ptr.get(), bytes);

                Timer<false> t;
                {
                    auto gpuBuffer = gpu.malloc<char, AccessMode::Read>(bytes);
                    auto mappedBuffer = gpuBuffer.map<AccessMode::Write>();
                    std::copy_n(ptr.get(), bytes, mappedBuffer.m_ptr);
                }
                std::cout << t.perSec(toMb(bytes)) << " MB/s\n";
            }
            catch (cl::Error const& err)
            {
                PrintFailureMessage("Testing <clEnqueueMapBuffer> failed: ", err);
                throw;
            }
        }

        /**
         * @brief Test the performance of copying data from host -> device
         */
        void CopyToDevice()
        {
#ifdef ANDROID
            const auto testBytes = { 4_kb, 1_mb,32_mb, 512_mb, 1_gb, 2_gb };
            const auto mapBytes = { 4_kb, 1_mb,32_mb, 512_mb, 1_gb, 2_gb };
#else
            const auto testBytes = { 4_kb, 1_mb,32_mb, 512_mb, 1_gb, 2_gb, 4_gb, 6_gb };
            const auto mapBytes = { 4_kb, 1_mb,32_mb, 512_mb, 1_gb, 2_gb};
#endif
            try {
                for (auto const bytes : testBytes)
                {
                    gpu.finish();
                    CopyHostPtr(bytes);
                }
            }catch(...){}
            try {
                for (auto const bytes : testBytes)
                {
                    gpu.finish();
                    WriteBuffer(bytes);
                }
            }catch(...){}
            try {
                for (auto const bytes : mapBytes)
                {
                    gpu.finish();
                    WriteMapBuffer(bytes);
                }
            }catch(...){}
            try {
                for (auto const bytes : testBytes)
                {
                    gpu.finish();
                    WriteBufferTotal(bytes);
                }
            }catch(...){}
            try {
                for (auto const bytes : mapBytes)
                {
                    gpu.finish();
                    WriteMapBufferTotal(bytes);
                }
            }catch(...){}
            try {
                gpu.finish();
            }catch(...){}
        }

        /**
         * @brief Test the performance of reading gpu data using clEnqueueReadBuffer
         * @param bytes Size for the test data to be copied
         */
        void ReadBuffer(size_t bytes)
        {
            try {
                std::cout << "Testing <clEnqueueReadBuffer> " << toMb(bytes) << " MB -> ";
                auto gpuBuffer = gpu.malloc<char, AccessMode::Write>(bytes);
                auto const ptr = std::make_unique<char[]>(bytes);
                /*generate dummy data*/
                gpu.enqueueKernel(gpu["TestRead"], std::make_tuple(gpuBuffer.getClBuffer()), { 0 }, { bytes });
                gpu.finish();

                Timer <false> t;
                {
                    gpuBuffer.copyTo(ptr.get(), bytes, true);
                    gpu.finish();
                }
                std::cout << t.perSec(toMb(bytes)) << " MB/s\n";
            }
            catch(cl::Error const& err)
            {
                PrintFailureMessage("Testing <clEnqueueReadBuffer> failed: ", err);
                throw;
            }
        }

        /**
         * @brief Test the performance of reading gpu data using clEnqueueMapBuffer
         * @param bytes Size for the test data to be copied
         */
        void ReadMapBuffer(size_t bytes)
        {
            try{
                std::cout << "Testing <clEnqueueMapBuffer> " << toMb(bytes) << " MB -> ";
                auto gpuBuffer = gpu.malloc<char, AccessMode::Write>(bytes);
                auto const ptr = std::make_unique<char[]>(bytes);
                /*generate dummy data*/
                gpu.enqueueKernel(gpu["TestRead"], std::make_tuple(gpuBuffer.getClBuffer()), { 0 }, { bytes });
                gpu.finish();

                Timer<false> t;
                {
                    auto mappedBuffer = gpuBuffer.map<AccessMode::Read>();
                    std::copy_n(mappedBuffer.m_ptr, bytes, ptr.get());
                }
                std::cout << t.perSec(toMb(bytes)) << " MB/s\n";
            }
            catch(cl::Error const& err)
            {
                PrintFailureMessage("Testing <clEnqueueMapBuffer> failed: ", err);
                throw;
            }
        }

        void CopyToHost()
        {
#ifdef ANDROID
            const auto testBytes = { 4_kb, 1_mb,32_mb, 512_mb, 768_mb };
            const auto mapBytes = { 4_kb, 1_mb,32_mb, 512_mb, 768_mb };
#else
            const auto testBytes = { 4_kb, 1_mb,32_mb, 512_mb, 1_gb, 2_gb }; //NVIDIA 1660Super does not go over 2GB ReadBuffer
            const auto mapBytes = { 4_kb, 1_mb,32_mb, 512_mb, 1_gb, 2_gb };
#endif

            try {
                for (auto const bytes : testBytes)
                {
                    gpu.finish();
                    ReadBuffer(bytes);
                }
            }
            catch(...)
            {}
            try {
                for (auto const bytes : mapBytes)
                {
                    gpu.finish();
                    ReadMapBuffer(bytes);
                }
            }
            catch(...)
            {}
            try {
                gpu.finish();
            } catch(...){}
        }

        void MapReadWrite()
        {
            
        }

        template<typename T>
        [[maybe_unused]]T ReadAsType(size_t bytes)
        {
            std::cout << "Read RAM as type: " << typeid(T).name();
            const auto count = bytes / sizeof(T);
            auto ptr = std::make_unique<T[]>(count);
            T result{};
            {
                Timer<false> t;
#ifndef ANDROID
                std::for_each_n(std::execution::par_unseq, ptr.get(), count, [](auto& element) {++element; });
#endif
                std::cout << " Speed: " << toGb(t.perSec(bytes)) << " GB/s \n";
            }
            return result;
        }

        void RamSpeed()
        {
            ReadAsType<unsigned char>(10_gb);
            ReadAsType<unsigned short>(10_gb);
            ReadAsType<unsigned int>(10_gb);
            ReadAsType<unsigned long long>(10_gb);
        }

        void DataTransfer()
        {
            std::cout << "///////////Host -> Device///////////\n";
            CopyToDevice();
            std::cout << "\n\n////////////Device -> Host///////////\n";
            CopyToHost();
        }

    }

    namespace Compilation
    {
#ifdef ANDROID
        static std::array dirIter
        {
            "AddTwo",
            "AddTwoTo",
            "DivideTwo",
            "DivideTwoTo",
            "MinusTwo",
            "MinusTwoTo",
            "MulTwo",
            "MulTwoTo",
            "SelfAdd",
            "SelfAddTo",
            "SelfDivide",
            "SelfDivideTo",
            "SelfMinus",
            "SelfMinusTo",
            "SelfMul",
            "SelfMulTo",
            "SumAll"
        };
        static std::array kernelIter
        {
            "AddTwo",
            "AddTwoTo",
            "DivideTwo",
            "DivideTwoTo",
            "MinusTwo",
            "MinusTwoTo",
            "MulTwo",
            "MulTwoTo",
            "SelfAdd",
            "SelfAddTo",
            "SelfDivide",
            "SelfDivideTo",
            "SelfMinus",
            "SelfMinusTo",
            "SelfMul",
            "SelfMulTo",
            "SumAll"
        };
#endif
        void SingleThread(bool saveKernel)
        {
            std::cout << "Testing <CompileSingleThread> -> ";
#ifdef ANDROID
            int count{};
            Timer<true> t;
            for (auto path : dirIter)
            {
                auto kernel = compiler.build(path, { CompileOption::Optimize::FastMath }, { CompileOption::Std::CL2_0 });
#ifdef DEBUG
                std::cout << "Kernel: <" << path << "> created\n";
#endif
                if(saveKernel)
                    compiler.saveKernel(path, kernel[0]);

                ++count;
            }
            std::cout << t.perSec(count) << " kernels /s\n";
#else

            std::filesystem::directory_iterator dirIter{ "./test" };
            int count{};
            Timer<true> t;
            for (auto&& entry : dirIter)
            {
                if (auto const& path = entry.path(); path.extension() == ".cl")
                {
                    auto kernel = compiler.build(path, { CompileOption::Optimize::FastMath }, { CompileOption::Std::CL2_0 });
#ifdef DEBUG
                    std::cout << "Kernel: <" << path << "> created\n";
#endif
                    if (saveKernel)
                        compiler.saveKernel(path, kernel[0]);

                    ++count;
                }
            }
            std::cout << t.perSec(count) << " kernels /s\n";
#endif
        }

        void MultiThreadWithAsync()
        {
            std::cout << "Testing <CompilingMultiThreadAsync> -> ";
#ifdef ANDROID
            std::vector<std::future<void>> buildFutures;
            buildFutures.reserve(50);
            int count{};
            Timer<true> t;

            for (auto path : dirIter)
            {
                    buildFutures.emplace_back(std::async(std::launch::async, [path, &count]()
                    {
                        auto kernels = compiler.build(path, { CompileOption::Optimize::FastMath }, { CompileOption::Std::CL2_0 });
#ifdef DEBUG
                        std::cout << "Kernel: <" << path << "> created\n";
#endif
                        ++count;
                    }));
            }
#else
            std::vector<std::future<void>> buildFutures;
            buildFutures.reserve(50);
            int count{};
            Timer<true> t;

            std::filesystem::directory_iterator dirIter{ "./test" };
            for (auto&& entry : dirIter)
            {
                if (auto path = entry.path(); path.extension() == ".cl")
                {
                    buildFutures.emplace_back(std::async(std::launch::async, [path = std::move(path), &count]()
                    {
                        auto kernels = compiler.build(path, { CompileOption::Optimize::FastMath }, { CompileOption::Std::CL2_0 });
#ifdef DEBUG
                        std::cout << "Kernel: <" << path << "> created\n";
#endif
                        ++count;
                    }));
                }
            }
#endif

            /*wait for all futures to finish */
            for (auto& future : buildFutures)
                future.wait();
            std::cout << t.perSec(count) << " kernels /s\n";
        }

        void MultiThreadWithThread()
        {
            std::cout << "Testing <CompilingMultiThread> -> ";
#ifdef ANDROID
            std::vector<std::thread> threads;
            threads.reserve(50);
            int count{};
            Timer<true> t;

            for (auto path : dirIter)
            {
                    threads.emplace_back([path, &count]()
                    {
                        auto kernels = compiler.build(path, { CompileOption::Optimize::FastMath }, { CompileOption::Std::CL2_0 });
#ifdef DEBUG
                        std::cout << "Kernel: <" << path << "> created\n";
#endif
                        ++count;
                    });
            }
#else
            std::vector<std::thread> threads;
            threads.reserve(50);
            int count{};
            Timer<true> t;

            std::filesystem::directory_iterator dirIter{ "./test" };
            for (auto&& entry : dirIter)
            {
                if (auto path = entry.path(); path.extension() == ".cl")
                {
                    threads.emplace_back([path = std::move(path), &count]()
                    {
                        auto kernels = compiler.build(path, { CompileOption::Optimize::FastMath }, { CompileOption::Std::CL2_0 });
#ifdef DEBUG
                        std::cout << "Kernel: <" << path << "> created\n";
#endif
                        ++count;
                    });
                }
            }
#endif

            /*wait for all threads to finish */
            for (auto& thread : threads)
                thread.join();
            std::cout << t.perSec(count) << " kernels /s\n";
        }

        void LoadFromBinary()
        {
            /*First build the kernels*/
            SingleThread(true);
#ifdef ANDROID
            /*Then load the kernels*/
            int count = {}; //reset count
            Timer<true> t;
            for (auto path : kernelIter)
            {
                    auto kernel = compiler.loadKernel(path, { devices.devices[0] });
#ifdef DEBUG

                    std::cout << "Kernel: <" << path << "> loaded\n";
#endif
                    ++count;
            }
            
#else

            /*Then load the kernels*/
            std::cout << "Testing <LoadingBinarySingleThread> -> ";
            std::filesystem::directory_iterator dirIter{ "./test" };    //reset dirIter
            int count = {}; //reset count
            Timer<true> t;
            for (auto&& entry : dirIter)
            {
                if (auto const& path = entry.path(); path.extension() == ".bin")
                {
                    auto kernel = compiler.loadKernel(path, { devices.devices[gpuIndex] });
#ifdef DEBUG
    #ifdef _WIN32
                    std::wcout << "Kernel: <" << path.stem().c_str() << "> loaded\n";   //In Win32, the c_str() is a wchar_t string
    #else
                    std::cout << "Kernel: <" << path.stem().c_str() << "> loaded\n";
    #endif
#endif
                    ++count;
                }
            }
#endif
            std::cout << "Loaded " << count << " kernels from binary "<< t.perSec(count)<<" kernels /s\n";
        }

        void MultiThreadLoadFromBinary()
        {
            /*First build the kernels*/
            SingleThread(true);

#ifdef ANDROID
            /*Then load the kernels*/
            int count = {}; //reset count
            std::vector<std::future<void>> buildFutures;
            buildFutures.reserve(count);
            Timer<true> t;


            for (auto path : kernelIter)
            {
                    buildFutures.emplace_back(std::async(std::launch::async, [fpath = path, &count]
                        {
                            auto kernel = compiler.loadKernel(fpath, { devices.devices[0] });
    #ifdef DEBUG
                            std::cout << "Kernel: <" << fpath << "> loaded\n";
    #endif
                            ++count;
                        }));
            }
#else

            /*Then load the kernels*/
            std::cout << "Testing <LoadingBinaryMultiThread> -> ";
            std::filesystem::directory_iterator dirIter{ "./test" };    //reset dirIter
            int count = {}; //reset count
            std::vector<std::future<void>> buildFutures;
            buildFutures.reserve(count);
            Timer<true> t;


            for (auto&& entry : dirIter)
            {
                if (auto path = entry.path(); path.extension() == ".bin")
                {
                    buildFutures.emplace_back(std::async(std::launch::async, [fpath = std::move(path), &count]
                        {
                            auto kernel = compiler.loadKernel(fpath, { devices.devices[gpuIndex] });
    #ifdef DEBUG
        #ifdef _WIN32
                            std::wcout << "Kernel: <" << fpath.stem().c_str() << "> loaded\n";   //In Win32, the c_str() is a wchar_t string
        #else
                            std::cout << "Kernel: <" << path.stem().c_str() << "> loaded\n";
        #endif
    #endif
                            ++count;
                        }));
                }
            }
#endif

            for (auto& future : buildFutures)
                future.wait();

            std::cout << "Loaded " << count << " kernels from binary " << t.perSec(count) << " kernels /s\n";
        }

        void Compilation()
        {
            SingleThread();
            MultiThreadWithAsync();

            MultiThreadWithThread();
            LoadFromBinary();
            MultiThreadLoadFromBinary();
        }

    }

    namespace Timing
    {
        static inline void PrintProfilingDiff(cl::Event const& timingEvent, std::chrono::steady_clock::duration scopedTimerTick)
        {
            auto const clTime = timingEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timingEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << "OpenCL profile result: " << clTime / 1000 << " ns. Diff = " << ::abs(std::chrono::duration_cast<std::chrono::microseconds>(scopedTimerTick).count(), clTime / 1000) << '\n';
        }

        void ProfileKernelExecution()
        {
            /*Create a new command_queue with profiling enabled*/
            cl::CommandQueue queue{ gpu.getCLContext(), CL_QUEUE_PROFILING_ENABLE };
            cl::Event timing;

            /*allocate buffer*/
            const auto size = 300'000;
            cl::Buffer resultBuffer{ gpu.getCLContext(), CL_MEM_WRITE_ONLY, size };

            /*Get kernel ready*/
            auto kernel = gpu["FindPrime"];
            kernel.setArg(0, resultBuffer);
            kernel.setArg(1, cl_ulong{ 3 });

            /*enqueue kernel*/
            std::chrono::steady_clock::duration duration{};
            {
                Timer<true> const t;
                queue.enqueueNDRangeKernel(kernel, { 0 }, { size }, cl::NullRange, {}, &timing);
                queue.finish();
                duration = t.getDuration();
            }
            /*get profiling info*/
            PrintProfilingDiff(timing, duration);
        }

        void ProfileDataTransfer()
        {
            /*Create a new command_queue with profiling enabled*/
            cl::CommandQueue queue{ gpu.getCLContext(), CL_QUEUE_PROFILING_ENABLE };
            cl::Event timing;

            /*allocate buffer*/
            constexpr auto size = 1_gb;
            cl::Buffer buffer{ gpu.getCLContext(), CL_MEM_READ_ONLY, size };
            auto const data = std::make_unique<char[]>(size);

            /*enqueue write buffer*/
            std::chrono::steady_clock::duration duration{};
            {
                Timer<true> const t;
                queue.enqueueWriteBuffer(buffer, true, 0, size, data.get(), {}, &timing);
                queue.finish();
                duration = t.getDuration();
            }
            /*get profiling info*/
            PrintProfilingDiff(timing, duration);
        }

    }

    namespace MemoryType
    {
        void UseHostPtr()
        {
            for(auto const size:{1_gb, 2_gb, 4_gb, 6_gb, 8_gb})
            {
                std::cout << "Using CL_MEM_USE_HOST_PTR to allocate " << toMb(size) << " MB buffer ";
                auto const ramUsageBefore = GetRamUsage();

                try {
                    auto hostPtr = std::make_unique<char[]>(size);
                    auto buffer = gpu.malloc<char, AccessMode::Read>(size, CL_MEM_USE_HOST_PTR, hostPtr.get());
                    auto const ramUsageAfter = GetRamUsage();

                    std::cout << "Ram usage = " << toMb(ramUsageAfter - ramUsageBefore) << " MB\n";
                }
                catch(cl::Error const& err)
                {
                    std::cerr << "Failed to allocate " << toMb(size) << " MB buffer";
                    PrintCLError(err);
                }
            }
            //Uses system ram
        }

        void AllocHostPtr()
        {
            for (auto const size : {1_gb, 2_gb, 4_gb, 6_gb, 8_gb})
            {
                std::cout << "Using CL_MEM_ALLOC_HOST_PTR to allocate " << toMb(size) << " MB buffer ";
                auto const ramUsageBefore = GetRamUsage();
                try {
                    auto buffer = gpu.malloc<char, AccessMode::Read>(size, CL_MEM_ALLOC_HOST_PTR);
                    auto const ramUsageAfter = GetRamUsage();

                    std::cout << "Ram usage = " << toMb(ramUsageAfter - ramUsageBefore) << " MB\n";
                }
                catch(cl::Error const& err)
                {
                    std::cerr << "Failed to allocate " << toMb(size) << " MB buffer";
                    PrintCLError(err);
                }
            }
            //uses nothing
        }

        void AllocHostPtrWithUseHostPtr()
        {
            for (auto const size : {1_gb, 2_gb, 4_gb, 6_gb, 8_gb})
            {
                std::cout << "Using CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_HOST_PTR to allocate " << toMb(size) << " buffer";
                auto const ramUsageBefore = GetRamUsage();
                auto buffer = gpu.malloc<char, AccessMode::Read>(size, CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_HOST_PTR);
                auto const ramUsageAfter = GetRamUsage();

                std::cout << "Ram usage = " << toMb(ramUsageAfter - ramUsageBefore) << " MB\n";
            }
        }


    }

    namespace Benchmark
    {
        constexpr size_t workGroupSize = 64;

        static inline size_t ceil(size_t nominator, size_t denominator)
        {
            return (nominator % denominator == 0 ? nominator / denominator : (nominator / denominator + 1));
        }

        namespace Reduction
        {
            auto makeData(size_t numElements)
            {
                static std::mt19937 eng{std::random_device{}()};
                static std::uniform_real_distribution<float> dist{ -1, 1 };
                auto buffer = std::make_unique<float[]>(numElements);
                std::generate(buffer.get(), buffer.get() + numElements, []() {return dist(eng); });
                return buffer;
            }

            void StdAccumulate(size_t numElements)
            {
                std::cout << "Testing <std::accumulate> with " << numElements << '\n';
                auto buffer = makeData(numElements);
                float result{};

                {
                    Timer<true> t;
                    result = std::accumulate(buffer.get(), buffer.get() + numElements, 0.0f);
                    std::cout << toGb(t.perSec(numElements * sizeof(float))) << " GB/s. Reduce result = " << result << '\n';
                }
            }

            void StdReduce(size_t numElements)
            {
                std::cout << "Testing <std::reduce> with " << numElements << '\n';
                auto buffer = makeData(numElements);
                float result{};

                {
                    Timer<true> t;
                    result = std::reduce(buffer.get(), buffer.get() + numElements, 0.0f);
                    std::cout << toGb(t.perSec(numElements * sizeof(float))) << " GB/s. Reduce result = " << result << '\n';
                }
            }

            void InterleavedAddressingImpl(const char* kernelName, size_t numElements)
            {
                auto const total = numElements;
                std::cout << "Testing <"<< kernelName <<"> with " << numElements << '\n';
                auto data = makeData(numElements);
                auto numWorkGroups = ceil(numElements, workGroupSize);


                auto inBuffer = gpu.malloc<float, AccessMode::ReadWrite>(numElements, data.get());
                auto outBuffer = gpu.malloc<float, AccessMode::ReadWrite>(numWorkGroups);

                auto kernel = gpu[kernelName];
                {
                    int round{};
                    Timer<true> t;
                    while (numElements >= workGroupSize)
                    {
                        gpu.enqueueKernel(
                            kernel,
                            std::make_tuple(inBuffer.getClBuffer(), outBuffer.getClBuffer(), std::make_tuple(sizeof(float) * workGroupSize, nullptr)),
                            { 0 },
                            { numElements },
                            { ::std::min(workGroupSize, numElements) }
                        );
                        std::swap(inBuffer, outBuffer);
                        numElements = numWorkGroups;
                        numWorkGroups = ceil(numElements, workGroupSize);
                        ++round;
                    }
                    gpu.finish();
                    std::cout << toGb(t.perSec(total * sizeof(float))) << " GB/s Round = " << round << "\n";
                }

                auto mappedResult = inBuffer.map<AccessMode::Read>();
                std::cout << "Reduce result: " << *mappedResult.m_ptr << '\n';
            }

            void InterleavedAddressingDivergent(size_t numElements)
            {
                InterleavedAddressingImpl("ReduceInterleaved", numElements);
            }
            void InterleavedAddressingNonDivergent(size_t numElements)
            {
                InterleavedAddressingImpl("ReduceInterleavedNonDivergent", numElements);
            }


            void SequentialAddressing(size_t numElements)
            {
                //auto const total = numElements;
                //std::cout << "Testing <SeqentialAddressing> with " << numElements << '\n';
                //auto data = makeData(numElements);
                //auto numWorkGroups = ceil(numElements, workGroupSize);


                //auto inBuffer = gpu.malloc<float, AccessMode::ReadWrite>(numElements, data.get());
                //auto outBuffer = gpu.malloc<float, AccessMode::ReadWrite>(numWorkGroups);

                //auto kernel = gpu["ReduceInterleaved"];
                //{
                //    int round{};
                //    Timer<true> t;
                //    while (numElements >= workGroupSize)
                //    {
                //        gpu.enqueueKernel(
                //            kernel,
                //            std::make_tuple(inBuffer.getClBuffer(), outBuffer.getClBuffer(), std::make_tuple(sizeof(float) * workGroupSize, nullptr)),
                //            { 0 },
                //            { numElements },
                //            { ::std::min(workGroupSize, numElements) }
                //        );
                //        std::swap(inBuffer, outBuffer);
                //        numElements = numWorkGroups;
                //        numWorkGroups = ceil(numElements, workGroupSize);
                //        ++round;
                //    }
                //    gpu.finish();
                //    std::cout << toGb(t.perSec(total * sizeof(float))) << " GB/s Round = " << round << "\n";
                //}

                //auto mappedResult = inBuffer.map<AccessMode::Read>();
                //std::cout << "Reduce result: " << *mappedResult.m_ptr << '\n';

                InterleavedAddressingImpl("ReduceSequential", numElements);
            }

            void FirstAddDuringLoad(size_t numElements)
            {
                auto const total = numElements;
                std::cout << "Testing <FirstAddDuringLoad> with " << numElements << '\n';
                auto data = makeData(numElements);
                auto numWorkGroups = ceil(numElements, workGroupSize);


                auto inBuffer = gpu.malloc<float, AccessMode::ReadWrite>(numElements, data.get());
                auto outBuffer = gpu.malloc<float, AccessMode::ReadWrite>(numWorkGroups);

                auto kernel = gpu["FirstAddDuringLoad"];
                {
                    int round{};
                    Timer<true> t;
                    while (numElements >= workGroupSize)
                    {
                        gpu.enqueueKernel(
                            kernel,
                            std::make_tuple(inBuffer.getClBuffer(), outBuffer.getClBuffer(), std::make_tuple(sizeof(float) * workGroupSize, nullptr)),
                            { 0 },
                            { numElements/2 },
                            { ::std::min(workGroupSize, numElements) }
                        );
                        std::swap(inBuffer, outBuffer);
                        numElements = numWorkGroups;
                        numWorkGroups = ceil(numElements, workGroupSize);
                        ++round;
                    }
                    gpu.finish();
                    std::cout << toGb(t.perSec(total * sizeof(float))) << " GB/s Round = " << round << "\n";
                }

                auto mappedResult = inBuffer.map<AccessMode::Read>();
                std::cout << "Reduce result: " << *mappedResult.m_ptr << '\n';
                
            }

            void Reduction()
            {
                try {
                    for (size_t size=1ull<<18; size<=(1ull<<26); size<<=1)
                    {
                        InterleavedAddressingDivergent(size);
                    }
                }catch(...){}
                try {
                    for (size_t size = 1ull << 18; size <= (1ull << 26); size <<= 1)
                    {
                        InterleavedAddressingNonDivergent(size);
                    }
                }
                catch (...) {}
                try {
                    for (size_t size = 1ull << 18; size <= (1ull << 26); size <<= 1)
                    {
                        SequentialAddressing(size);
                    }
                }catch(...){}
                try {
                    for (size_t size = 1ull << 18; size <= (1ull << 26); size <<= 1)
                    {
                        FirstAddDuringLoad(size);
                    }
                }catch(...){}
            }
        }

        namespace MatrixMultiplication
        {
            void NaiveCPU(size_t size)
            {
                std::cout << "Testing <NaiveMulCPU> with " << size << " x " << size << '\n';
                auto a = Matrix::make_random_matrix(size, size);
                auto b = Matrix::make_random_matrix(size, size);
                {
                    Timer<true> t;
                    NaiveCPUMul(a, b);
                    std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
                }
            }
            void TransposedCPU(size_t size)
            {
                std::cout << "Testing <TransposedCPU> with " << size << " x " << size << '\n';
                auto a = Matrix::make_random_matrix(size, size);
                auto b = Matrix::make_random_matrix(size, size);
                {
                    Timer<true> t;
                    auto b_T = b.transpose();
                    TransposedCPUMul(a, b_T);
                    std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
                }
            }

            /**
             * @brief The naive matrix multiplication implementation
             */
            void Naive(size_t size)
            {
                std::cout << "Testing <NaiveMul> with " << size << " x " << size << '\n';
                auto a = Matrix::make_random_matrix(size, size);
                auto b = Matrix::make_random_matrix(size, size);


                Matrix result(size, size, Matrix::NoAlloc{});

                /*command*/
                auto a_buf = gpu.malloc<float, AccessMode::Read>(a.size(), a.data);
                auto b_buf = gpu.malloc<float, AccessMode::Read>(b.size(), b.data);
                auto result_buf = gpu.malloc<float, AccessMode::Write>(result.size());


                gpu.enqueueKernel(gpu["NaiveMul"], std::make_tuple(a_buf.getClBuffer(), b_buf.getClBuffer(), result_buf.getClBuffer()), {}, { size, size });
                {
                    Timer<true> t;
#ifdef DEBUG
                    auto mappedResult = result_buf.map<AccessMode::Read>();
                    result.data = mappedResult.m_ptr;
#else
                    gpu.finish();
                    std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
#endif
                }
            }

            /**
             * @brief Transpose the right hand side matrix by CPU and do the matrix multiplication on GPU
             */
            void TransposeByCPU(size_t size)
            {
                std::cout << "Testing <TransposedByCPU> with " << size << " x " << size << '\n';
                auto a = Matrix::make_random_matrix(size, size);
                auto b = Matrix::make_random_matrix(size, size);
                auto bT = b.transpose();


                Matrix result(size, size, Matrix::NoAlloc{});

                auto a_buf = gpu.malloc<float, AccessMode::Read>(a.size(), a.data);
                auto b_buf = gpu.malloc<float, AccessMode::Read>(b.size(), b.data);
                auto result_buf = gpu.malloc<float, AccessMode::Write>(result.size());

                gpu.enqueueKernel(gpu["TransposedMul"], std::make_tuple(a_buf.getClBuffer(), b_buf.getClBuffer(), result_buf.getClBuffer()), {}, { size, size });
                {
                    Timer<true> t;
#ifdef DEBUG
                    auto mappedResult = result_buf.map<AccessMode::Read>();
                    result.data = mappedResult.m_ptr;
#else
                    gpu.finish();
                    std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
#endif
                }
            }

            /**
             * @brief Directly copy the right hand side matrix -> GPU and do the transpose, then do the matrix multiplication
             */
            void TransposeByGPU(size_t size)
            {
                std::cout << "Testing <TransposeByGPU> with " << size << " x " << size << '\n';
                auto a = Matrix::make_random_matrix(size, size);
                auto b = Matrix::make_random_matrix(size, size);

                auto a_buf = gpu.malloc<float, AccessMode::Read>(b.size(), a.data);
                auto b_buf = gpu.malloc<float, AccessMode::Read>(b.size(), b.data);
                auto b_T_buf = gpu.malloc<float, AccessMode::ReadWrite>(b.size());

                Matrix result{ size, size, Matrix::NoAlloc{} };
                auto transposeKernel = gpu["Transpose"];
                auto transposeMulKernel = gpu["TransposedMul"];
                {
                    Timer<true> t;
                    gpu.enqueueKernel(std::move(transposeKernel), std::make_tuple(b_buf.getClBuffer(), b_T_buf.getClBuffer()), {}, { size, size });
                    auto result_buf = gpu.malloc<float, AccessMode::Write>(result.size());
                    gpu.enqueueKernel(std::move(transposeMulKernel), std::make_tuple(a_buf.getClBuffer(), b_T_buf.getClBuffer(), result_buf.getClBuffer()), {}, { size, size });
#ifdef DEBUG
                    auto mappedResult = result_buf.map<AccessMode::Read>();
                    result.data = mappedResult.m_ptr;
#else
                    gpu.finish();
                    std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
#endif

                }
            }

            
            /**
             * @brief Use tiled matrix block multiplication to make use of local memory
             */
            void UseLocalMemory(size_t size)
            {
                constexpr size_t block_dim = 8;
                std::cout << "Testing <BlockMul> with " << size << " x " << size << '\n';
                auto a = Matrix::make_test_matrix(size, size);
                auto b = Matrix::make_test_matrix(size, size);

                auto a_buf = gpu.malloc<float, AccessMode::Read>(b.size(), a.data);
                auto b_buf = gpu.malloc<float, AccessMode::Read>(b.size(), b.data);
                auto result_buf = gpu.malloc<float, AccessMode::Write>(b.size());

                Matrix result{ size, size, Matrix::NoAlloc{} };

                auto const localMemSize = block_dim*block_dim * sizeof(float);
               
                gpu.enqueueKernel(gpu["BlockMul"], std::make_tuple(a_buf.getClBuffer(), b_buf.getClBuffer(), std::make_tuple(localMemSize, nullptr), std::make_tuple(localMemSize, nullptr), result_buf.getClBuffer()), {}, { size, size }, { block_dim, block_dim });
                {
                    Timer<true> t;

#ifdef DEBUG
                    auto mappedResult = result_buf.map<AccessMode::Read>();
                    result.data = mappedResult.m_ptr;

#else
                    gpu.finish();
                    std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
#endif
                }
            }

            void UseNonConstantMemory(size_t size)
            {
                constexpr size_t block_dim = 8;
                std::cout << "Testing <BlockMulNonConstant> with " << size << " x " << size << '\n';
                auto a = Matrix::make_test_matrix(size, size);
                auto b = Matrix::make_test_matrix(size, size);

                auto a_buf = gpu.malloc<float, AccessMode::Read>(b.size(), a.data);
                auto b_buf = gpu.malloc<float, AccessMode::Read>(b.size(), b.data);
                auto result_buf = gpu.malloc<float, AccessMode::Write>(b.size());

                Matrix result{ size, size, Matrix::NoAlloc{} };

                auto const localMemSize = block_dim * block_dim * sizeof(float);

                gpu.enqueueKernel(gpu["BlockMulNonConstant"], std::make_tuple(a_buf.getClBuffer(), b_buf.getClBuffer(), std::make_tuple(localMemSize, nullptr), std::make_tuple(localMemSize, nullptr), result_buf.getClBuffer()), {}, { size, size }, { block_dim, block_dim });
                {
                    Timer<true> t;

#ifdef DEBUG
                    auto mappedResult = result_buf.map<AccessMode::Read>();
                    result.data = mappedResult.m_ptr;

#else
                    gpu.finish();
                    std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
#endif
                }
            }

            void UnrolledMul(size_t size)
            {
                constexpr size_t block_dim = 8;
                std::cout << "Testing <UnrolledBlockMul> with " << size << " x " << size << '\n';
                auto a = Matrix::make_test_matrix(size, size);
                auto b = Matrix::make_test_matrix(size, size);

                auto a_buf = gpu.malloc<float, AccessMode::Read>(b.size(), a.data);
                auto b_buf = gpu.malloc<float, AccessMode::Read>(b.size(), b.data);
                auto result_buf = gpu.malloc<float, AccessMode::Write>(b.size());

                Matrix result{ size, size, Matrix::NoAlloc{} };

                auto const localMemSize = block_dim * block_dim * sizeof(float);

                gpu.enqueueKernel(gpu["UnrolledMul"], std::make_tuple(a_buf.getClBuffer(), b_buf.getClBuffer(), std::make_tuple(localMemSize, nullptr), std::make_tuple(localMemSize, nullptr), result_buf.getClBuffer()), {}, { size, size }, { block_dim, block_dim });
                {
                    Timer<true> t;

#ifdef DEBUG
                    auto mappedResult = result_buf.map<AccessMode::Read>();
                    result.data = mappedResult.m_ptr;

#else
                    gpu.finish();
                    std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
#endif
                }
            }


            /**
             * @brief Use row-block-row-major ordering
             */
            void RowBlockRowMajorOrdering(size_t size)
            {
                constexpr size_t block_dim = 8;
                std::cout << "Testing <RowBlockRowMajorMul> with " << size << " x " << size << '\n';
                auto a = Matrix::make_test_matrix(size, size);
                auto b = Matrix::make_test_matrix(size, size);

                auto a_buf = gpu.malloc<float, AccessMode::Read>(b.size(), a.data);
                auto b_buf = gpu.malloc<float, AccessMode::Read>(b.size(), b.data);
                auto result_buf = gpu.malloc<float, AccessMode::Write>(b.size());

                Matrix result{ size, size, Matrix::NoAlloc{} };
                auto const localMemSize = block_dim * block_dim * sizeof(float);
                gpu.enqueueKernel(gpu["RowBlockRowMajorMul"], std::make_tuple(a_buf.getClBuffer(), b_buf.getClBuffer(), std::make_tuple(localMemSize, nullptr), std::make_tuple(localMemSize, nullptr), result_buf.getClBuffer()), {}, { size, size }, { block_dim, block_dim });
                {
                    Timer<true> t;
#ifdef DEBUG
                    auto mappedResult = result_buf.map<AccessMode::Read>();
                    result.data = mappedResult.m_ptr;

#else
                    gpu.finish();
                    std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
#endif
                }
            }




            void MoreWork(size_t size)
            {

                constexpr size_t block_dim = 8;
                try {
                    std::cout << "Testing <UseConstantMemory> with " << size << " x " << size << '\n';
                    auto a = Matrix::make_test_matrix(size, size);
                    auto b = Matrix::make_test_matrix(size, size);

                    auto a_buf = gpu.malloc<float, AccessMode::Read>(b.size(), a.data);
                    auto b_buf = gpu.malloc<float, AccessMode::Read>(b.size(), b.data);
                    auto result_buf = gpu.malloc<float, AccessMode::Write>(b.size());

                    Matrix result{ size, size, Matrix::NoAlloc{} };
                    gpu.enqueueKernel(gpu["MoreWorkMul"], std::make_tuple(a_buf.getClBuffer(), b_buf.getClBuffer(), result_buf.getClBuffer()), {}, { size, size }, { block_dim, block_dim });
                    {
                        Timer<true> t;
#ifdef DEBUG
                        auto mappedResult = result_buf.map<AccessMode::Read>();
                        result.data = mappedResult.m_ptr;

#else
                        gpu.finish();
                        std::cout << toGb(t.perSec(2 * pow(size, 3))) << " GFlops\n";
#endif
                    }
                }
                catch (cl::Error const& err)
                {
                    PrintFailureMessage("This gpu may not support big constant memory.", err);
                }
            }

            /**
             * @brief Test different methods of matrix multiplication
             */
            void MatrixMultiplication()
            {
                auto const sizes = { 128, 256, 512, 1024, 2048};
                for (const auto size :sizes)
                {
                    Naive(size);
                }
                for (const auto size : sizes)
                {
                    TransposeByCPU(size);
                    gpu.finish();
                }
                //for (const auto size : sizes)
                //{
                //    TransposeByGPU(size);
                //}
                for (const auto size : sizes)
                {
                    UseLocalMemory(size);
                }
                for(const auto size:sizes)
                {
                    UnrolledMul(size);
                }
                for (const auto size : sizes)
                {
                    UseNonConstantMemory(size);
                }
                for(const auto size:sizes)
                {
                    RowBlockRowMajorOrdering(size);
                }
                for(const auto size:sizes)
                {
                    MoreWork(size);
                }
            }
            void MatrixMultiplicationCPU()
            {
                auto const sizes = { 128, 256, 512, 1024, 2048 };
                for(auto size:sizes)
                {
                    //NaiveCPU(size);
                }
                for(auto size:sizes)
                {
                    TransposedCPU(size);
                }
            }
        }

        namespace Convolution
        {
            template<int filterSize, int channels>
            void NaiveImpl (size_t pixel)
            {
                std::cout << "Testing <NaiveConv> with " << pixel << " x " << pixel <<"channel = "<<channels <<" with filter = "<<filterSize<< '\n';
                auto filter = Filter<filterSize, channels>::makeFilter();
                Image<channels> inputImage{ pixel+filter.halfSize()*2, pixel+filter.halfSize()*2 };
                Image<channels> outputImage{ pixel, pixel, NoAlloc{} };

                auto inputBuf = gpu.malloc<unsigned char, AccessMode::Read>(inputImage.size(), inputImage.data);
                auto outputBuf = gpu.malloc<unsigned char, AccessMode::Write>(outputImage.size());

                auto const halfFilterSizeStr = std::to_string(filter.halfSize());
                auto const channelsStr = std::to_string(filter.channel());
                auto kernels = compiler.build("NaiveConv", 
                    { 
                        CompileOption::Optimize::FastMath,
                        CompileOption::Macro{"HALF_FILTER_SIZE", halfFilterSizeStr.c_str()},
                        CompileOption::Macro{"CHANNELS", channelsStr.c_str()}
                    }, { CompileOption::Std::CL2_0 });

                {
                    Timer<false> t;
                    gpu.enqueueKernel(
                        kernels[0],
                        std::forward_as_tuple(inputBuf.getClBuffer(), outputBuf.getClBuffer(), filter.data),
                        {},
                        { pixel, pixel }
                    );
                    gpu.finish();
                    std::cout << toGb(t.perSec(filter.area() * inputImage.size() * channels)) << " GFlops\n";
                }
            }

            template<int filterSize, int channels>
            void NaiveCPUConv(size_t pixel)
            {
                std::cout << "Testing <NaiveConvCPU> with " << pixel << " x " << pixel << "channel = " << channels << " with filter = " << filterSize << '\n';
                auto filter = Filter<filterSize, channels>::makeFilter();
                Image<channels> inputImage{ pixel + filter.halfSize() * 2, pixel + filter.halfSize() * 2 };
                {
                    Timer<true> t;
                    NaiveCPU(inputImage, filter);
                    std::cout << toGb(t.perSec(filter.area() * inputImage.size() * channels)) << " GFlops\n";
                }
            }

            void Naive(size_t pixel)
            {
                NaiveImpl<3, 1>(pixel);
                NaiveImpl<5, 1>(pixel);
                NaiveImpl<7, 1>(pixel);
                NaiveImpl<9, 1>(pixel);
                NaiveCPUConv<3, 1>(pixel);
                NaiveCPUConv<5, 1>(pixel);
                NaiveCPUConv<7, 1>(pixel);
                NaiveCPUConv<9, 1>(pixel);
            }

            template<int filterSize, int channels>
            void LoopUnrollImpl(size_t pixel, std::vector<cl::Kernel> const& kernels)
            {
                std::cout << "Testing <UnrolledConv> with " << pixel << " x " << pixel << "channel = " << channels << " with filter = " << filterSize << '\n';
                auto filter = Filter<filterSize, 1>::makeFilter();
                Image<channels> inputImage{ pixel + filter.halfSize() * 2, pixel + filter.halfSize() * 2 };
                Image<channels> outputImage{ pixel, pixel, NoAlloc{} };

                auto inputBuf = gpu.malloc<unsigned char, AccessMode::Read>(inputImage.size(), inputImage.data);
                auto outputBuf = gpu.malloc<unsigned char, AccessMode::Write>(outputImage.size());

                auto const halfFilterSizeStr = std::to_string(filter.halfSize());
                auto const channelsStr = std::to_string(filter.channel());


                gpu.enqueueKernel(kernels[(filterSize - 3) / 2], std::forward_as_tuple(inputBuf.getClBuffer(), outputBuf.getClBuffer(), filter.data), {}, { pixel, pixel });

                {
                    Timer<false> t;
                    gpu.finish();
                    std::cout << toGb(t.perSec(filter.area() * inputImage.size() * channels)) << " GFlops\n";
                }
            }

            void LoopUnroll(size_t pixel)
            {
                auto kernels = compiler.build("UnrolledConv",
                    {
                        CompileOption::Optimize::FastMath,
                        CompileOption::Macro{"CHANNELS", "1"}
                    }, { CompileOption::Std::CL2_0 });

                LoopUnrollImpl<3, 1>(pixel, kernels);
                LoopUnrollImpl<5, 1>(pixel, kernels);
            }

            template<int filterSize, int channels>
            void GroupedConvImpl(size_t pixel)
            {
                std::cout << "Testing <GroupedConv> with " << pixel << " x " << pixel << "channel = " << channels << " with filter = " << filterSize << '\n';
                auto filter = Filter<filterSize, channels>::makeFilter();
                Image<channels> inputImage{ pixel + filter.halfSize() * 2, pixel + filter.halfSize() * 2 };
                Image<channels> outputImage{ pixel, pixel, NoAlloc{} };

                auto inputBuf = gpu.malloc<unsigned char, AccessMode::Read>(inputImage.size(), inputImage.data);
                auto outputBuf = gpu.malloc<unsigned char, AccessMode::Write>(outputImage.size());

                auto const halfFilterSizeStr = std::to_string(filter.halfSize());
                auto const channelsStr = std::to_string(filter.channel());
                auto kernels = compiler.build("GroupedConv",
                    {
                        CompileOption::Optimize::FastMath,
                        CompileOption::Macro{"HALF_FILTER_SIZE", halfFilterSizeStr.c_str()},
                        CompileOption::Macro{"CHANNELS", channelsStr.c_str()}
                    }, { CompileOption::Std::CL2_0 });

                const auto localDim = static_cast<size_t>(sqrt(workGroupSize));

                {
                    Timer<false> t;
                    gpu.enqueueKernel(
                        kernels[0],
                        std::forward_as_tuple(inputBuf.getClBuffer(), outputBuf.getClBuffer(), filter.data, std::make_tuple(channels*localDim*localDim*sizeof(float), nullptr)),
                        {},
                        { pixel, pixel },
                        { localDim, localDim }
                    );
                    gpu.finish();
                    std::cout << toGb(t.perSec(filter.area() * inputImage.size() * channels)) << " GFlops\n";
                }
            }

            void GroupedConv(size_t pixel)
            {
                GroupedConvImpl<3, 1>(pixel);
                GroupedConvImpl<5, 1>(pixel);
                GroupedConvImpl<7, 1>(pixel);
                GroupedConvImpl<9, 1>(pixel);
            }

            void Convolution()
            {
                Naive(4096);
                //LoopUnroll(8192);
                GroupedConv(4096);
            }
        }
    }
}


