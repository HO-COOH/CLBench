#include <CL/opencl.hpp>
#include "KernelInfo.hpp"

#include "GPU.h"

//TODO: here
KernelInfo::KernelInfo(cl::Kernel const& kernel) :KernelInfo(kernel, gpu.getCLDevice())
{
}

KernelInfo::KernelInfo(cl::Kernel const& kernel, cl::Device const& device)
    : localMemSize(kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device)),
    workGroupSize(kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device)),
    preferredWorkGroupSizeMultiple(kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)),
    privateMemSize(kernel.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device)),
    device(device)
{
}

bool KernelInfo::checkKernel() const
{
    return localMemSize >= device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
}