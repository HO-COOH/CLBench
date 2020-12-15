/*****************************************************************//**
 * \file   KernelInfo.hpp
 * \brief  Check the validity of OpenCL kernel
 * 
 * \author Wenhao Li
 * \date   October 2020
 *********************************************************************/
#pragma once

namespace cl {
    class Kernel;
    class Device;
}

struct KernelInfo
{
    const unsigned long long localMemSize;
    const unsigned long long workGroupSize;
    const unsigned long long preferredWorkGroupSizeMultiple;
    const unsigned long long privateMemSize;
    cl::Device const& device;
    KernelInfo(cl::Kernel const& kernel);
    KernelInfo(cl::Kernel const& kernel, cl::Device const& device);
    [[nodiscard]] bool checkKernel() const;
};