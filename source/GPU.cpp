#include "GPU.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

static auto GetCLDevice()
{
    /*Get CL devices*/
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::vector<cl::Device> devices;
    for (auto& platform : platforms)
    {
        std::vector<cl::Device> platformDevice;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &platformDevice);
        std::copy(platformDevice.cbegin(), platformDevice.cend(), std::back_inserter(devices));
    }
#ifdef DEBUG
    for(auto const& device:devices)
    {
        std::cout << "GPU: " << device.getInfo<CL_DEVICE_NAME>() << " found\n";
    }
#endif
    return devices;
}

static auto GetGPU(std::vector<cl::Device> const& devices)
{
    std::vector<ComputeDevice> gpus;
    gpus.reserve(devices.size());
    try {
        for (auto const& device : devices)
            gpus.emplace_back(device);
    }
    catch(cl::Error& err) {
        //The exception of "clCreateCommandQueueWithProperties" will happen there is a device does NOT
        //support the target OpenCL standard which is defined by the macro CL_HPP_MINIMUM_OPENCL_VERSION
        //but because the emplace_back has strong exception guarantee, the problematic device will not be added 
        std::cerr << "GetGPU() error: " << err.what() << " Code: " << err.err() << " -> " << GetErrorDescription(err.err()) << '\n';
    }
    //However we need to ensure there is at least 1 device usable
#ifdef DEBUG
    assert(!gpus.empty());
#endif
    return gpus;
}

static auto DeviceFactory()
{
    auto clDevices = GetCLDevice();
    return Devices{
        GetGPU(clDevices),
        clDevices,
        //cl::Context{clDevices}
    };
}


ComputeDevice::ComputeDevice(cl::Device device)
    :cl::Device{ std::move(device) },
    cl::Context{static_cast<cl::Device&>(*this)},
    cl::CommandQueue{ static_cast<cl::Context const&>(*this), static_cast<cl::Device&>(*this)}
{
}

ComputeDevice::~ComputeDevice()
{
    try {
        finish();
    }
    catch (cl::Error const& e) {
        std::cerr << "An error happens in finishing the queue. Code " << e.err() << ':' << e.what() << '\n';
    }
    catch (...) {
        std::cerr << "An unknown error happens in finishing the queue.\n";
    }
}

void ComputeDevice::finish()
{
    cl::CommandQueue::finish();
}


Devices devices=DeviceFactory();
ComputeDevice& gpu = devices.gpus[gpuIndex];
//TODO: The compiler instance is using the default gpu's context

Compiler compiler{devices.gpus[gpuIndex].getCLContext()};
unsigned Initializer::count;
void Initializer::init()
{
    //devices.devices = GetDevice();
    //std::copy(devices.devices.begin(), devices.devices.end(), std::back_inserter(devices.gpus));
    //devices.context = cl::Context{ devices.devices };
    //puts("GPU init");
}

void Initializer::cleanUp()
{
    puts("Waiting for GPU to finish.");
    for (auto& gpu : devices.gpus)
        gpu.~ComputeDevice();
    puts("GPU finish");
}

Initializer::Initializer()
{
    if (count++ == 0)
        init();
}

Initializer::~Initializer()
{
    if (--count == 0)
        cleanUp();
}




cl::Kernel& ComputeDevice::operator[](const char* kernelName)
{
    //First find the whether the required kernel is already compiled
    if (auto iter = kernels.find(kernelName); iter != kernels.end())
        return iter->second;

    //Not found, so compile it
    for (auto&& compiledKernels : compiler.build(kernelName, { CompileOption::Optimize::FastMath }, { CompileOption::Std::CL2_0 }))
    {
        kernels.insert({ kernelName, std::move(compiledKernels) });
#ifdef DEBUG
        std::cout << "Kernel: <" << kernelName << "> created\n";
#endif
    }
    return kernels[kernelName];
}


#include "when.hpp"
Vendor ComputeDevice::getVendor() const
{
    return when(gpu.getCLDevice().getInfo<CL_DEVICE_VENDOR>(),
        "AMD",      Vendor::AMD,
        "NVIDIA Corporation",   Vendor::NVIDIA,
        "QUALCOMM",             Vendor::Qualcomm,
        "Intel(R) Corporation", Vendor::Intel,
        Else(),                 Vendor::Other
    );
}
