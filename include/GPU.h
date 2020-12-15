/*****************************************************************//**
 * \file   GPU.h
 * \brief  OpenCL backend library
 * 
 * \author Peter
 * \date   September 2020
 *********************************************************************/
#pragma once

#include <CL/opencl.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include "Compiler.h"
#include "MappedBuffer.h"

enum class Vendor { AMD, NVIDIA, Intel, Qualcomm, Other };
constexpr static inline auto gpuIndex = 0;


struct ComputeDevice:private cl::Device, private cl::Context, private cl::CommandQueue
{
private:

    std::unordered_map<std::string, cl::Kernel> kernels;

    template<typename Tuple>
    static void setArgs(cl::Kernel& kernel, Tuple const& args);

    auto& getCLDevice() { return static_cast<cl::Device&>(*this); }
    auto& getCLDevice() const { return static_cast<cl::Device const&>(*this); }
public:
    auto& getCLQueue() { return static_cast<cl::CommandQueue&>(*this); }
    auto& getCLContext() { return static_cast<cl::Context&>(*this); }
    ComputeDevice(cl::Device device);
    ~ComputeDevice();


    /**
     * @brief Block and wait for all the command in the queue to finish
     */
    void finish();


    template<typename T>
    auto mallocRead(size_t count)
    {
        return Buffer<T>{count, getCLContext(), getCLQueue(), AccessMode::Read};
    }

    template<typename T>
    auto mallocWrite(size_t count)
    {
        return Buffer<T>{count, getCLContext(), getCLQueue(), AccessMode::Write};
    }

    template<typename T>
    auto mallocReadWrite(size_t count)
    {
        return Buffer<T>{count, getCLContext(), getCLQueue(), AccessMode::ReadWrite};
    }

    template<typename T>
    auto mallocRead(size_t count, T const* const data)
    {
        return Buffer<T>{count, getCLContext(), data, getCLQueue(), AccessMode::Read};
    }

    template<typename T>
    auto mallocWrite(size_t count, T const* const data)
    {
        return Buffer<T>{count, getCLContext(), data, getCLQueue(), AccessMode::Write};
    }

    template<typename T>
    auto mallocReadWrite(size_t count, T const* const data)
    {
        return Buffer<T>{count, getCLContext(), data, getCLQueue(), AccessMode::ReadWrite};
    }

    template<typename T, AccessMode mode>
    auto malloc(size_t count)
    {
        return Buffer<T>{count, getCLContext(), getCLQueue(), mode};
    }

    template<typename T, AccessMode mode>
    auto malloc(size_t count, T const* const data)
    {
        return Buffer<T>{count, getCLContext(), data, getCLQueue(), mode};
    }

    template<typename T, AccessMode mode>
    auto malloc(size_t count, int extraFlags, T * const data = nullptr)
    {
        return Buffer<T>{count, getCLContext(), getCLQueue(), mode, extraFlags, data};
    }



    /**
     * @brief Enqueue kernel with tuple of kernel arguments
     * @details
     * Enabling non-uniform work-groups requires a kernel to be compiled with the -cl-std=CL2.0 flag
     * and without the -cl-uniform-work-group-size flag.
     * If the program was created using clLinkProgram and any of the linked programs were compiled in a way that only supports uniform work-group sizes, the linked program only supports uniform work group sizes.
     * If local_work_size is specified and the OpenCL kernel is compiled without non-uniform work-groups enabled, the values specified in global_work_size[0], …​ global_work_size[work_dim - 1] must be evenly divisible by the corresponding values specified in local_work_ size[0], …​ local_work_size[work_dim – 1].
     */
    template<typename Tuple>
    void enqueueKernel(
        cl::Kernel kernel,
        Tuple&& args,
        const cl::NDRange& offset,
        const cl::NDRange& global,
        const cl::NDRange& local = cl::NullRange);

    cl::Kernel& operator[](const char* kernelName);

    [[nodiscard]]Vendor getVendor() const;

    /*delete all other special member functions */
    ComputeDevice(ComputeDevice const&) = delete;
    ComputeDevice(ComputeDevice &&) = default;
    ComputeDevice& operator=(ComputeDevice const&) = delete;
    ComputeDevice& operator=(ComputeDevice&&) = default;

    friend struct KernelInfo;
    friend class Compiler;
    template<typename T>
    friend class Buffer;
};

struct Devices
{
    std::vector<ComputeDevice> gpus;    //all gpus on the system
    std::vector<cl::Device> devices;     //just to satisfy the compiler
    cl::Context context;                //a context is consists of multiple gpus
};


extern ComputeDevice& gpu;          //the default gpu
extern Devices devices;             //all the gpus with context
extern Compiler compiler;           //the global compiler


class Initializer
{
    static unsigned count;
    static void init();
    static void cleanUp();
public:
    Initializer();
    ~Initializer();

    /*delete every other constructor and assignment */
    Initializer(Initializer const&) = delete;
    Initializer(Initializer&&) = delete;
    Initializer& operator=(Initializer const&) = delete;
    Initializer& operator=(Initializer&&) = delete;
};
static Initializer init;

struct ArgSetter
{
    template<typename T>
    void operator()(cl_int i, cl::Kernel& kernel, T arg)
    {
        kernel.setArg(i, arg);
    }

    template<typename ...T>
    void operator()(cl_int i, cl::Kernel& kernel, std::tuple<T...> const& args)
    {
        std::apply([&kernel, i](auto&&... args)
        {
            kernel.setArg(i, args...);
        }, args);
    }
};
template<typename Tuple>
void ComputeDevice::setArgs(cl::Kernel& kernel, Tuple const& args)
{
    cl_int i = 0;

    std::apply([&i, &kernel](auto&&... arg)
    {
        (..., ArgSetter{}(i++, kernel, arg));
    }, args);
}

template<typename Tuple>
void ComputeDevice::enqueueKernel(cl::Kernel kernel, Tuple&& args, const cl::NDRange& offset, const cl::NDRange& global, const cl::NDRange& local)
{
    setArgs(kernel, args);
    enqueueNDRangeKernel(kernel, offset, global, local);
    flush();
}

