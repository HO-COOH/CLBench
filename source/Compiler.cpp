#include "Compiler.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <future>


static auto GetSource(std::string const& fileName)
{
    std::ifstream f{ fileName };
    if (!f.is_open())
    {
        std::cerr << "Cannot open file: " << fileName << '\n';
        throw std::runtime_error{ "Cannot open file" };
    }
    return std::string{ std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{} };
}

std::vector<cl::Kernel> Compiler::build(const char* kernelName) const
{
    cl::Program program{ context, GetSource(std::string{kernelName} + ".cl") };
    try {
        program.build();
    }
    catch (cl::Error& err)
    {
        std::cerr << "Build program failed. Code: " << err.err() << '\n';
        const auto buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for (const auto& log : buildLog)
            std::cerr << log.first.getInfo<CL_DEVICE_NAME>() << ":\t" << log.second << '\n';
        throw std::runtime_error{ "Build program failed" };
    }
    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);
    std::cerr << "Kernel creation success!\n";
    return kernels;
}

std::vector<cl::Kernel> Compiler::build(const char* kernelName, CompileOption const& essentialFlag) const
{
    cl::Program program{ context, GetSource(std::string{kernelName} + ".cl") };
    try {
        program.build(essentialFlag.option.c_str());
    }
    catch (cl::Error& err)
    {
        std::cerr << "Build program failed. Code: " << err.err() << '\n';
        const auto buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for (const auto& log : buildLog)
            std::cerr << log.first.getInfo<CL_DEVICE_NAME>() << ":\t" << log.second << '\n';
        throw std::runtime_error{ "Build program failed" };
    }
    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);
    std::cerr << "Kernel creation success!\n";
    return kernels;
}

void Compiler::buildAll() const
{
}

#ifndef ANDROID
void Compiler::buildAll(std::unordered_map<std::string, cl::Kernel>& storage, CompileOption const& essentialFlag,
    CompileOption const& otherFlags) const
{
    std::mutex storageMutex;

    std::vector<std::future<void>> buildFutures;
    buildFutures.reserve(50);

    /*loop through all the .cl file*/
    std::filesystem::directory_iterator dirIter{ "." };
    for(auto&& entry:dirIter)
    {
        if(auto path=entry.path(); path.extension()==".cl")
        {
            buildFutures.emplace_back(std::async(std::launch::async, [&path, &storageMutex, &storage, &essentialFlag, &otherFlags, this]()mutable
            {
                auto name = path.stem().string();
                auto kernels=build(name.c_str(), essentialFlag, otherFlags);
                {
                    std::lock_guard lock{ storageMutex };
                    for(auto& kernel:kernels)
                        storage.insert({ std::move(name), std::move(kernel) });
                }
            }));
        }
    }

    /*wait for all futures to finish */
    for (auto& future : buildFutures)
        future.wait();
}
#endif

void Compiler::buildAll(CompileOption const& essentialFlag) const
{
}

std::vector<cl::Kernel> Compiler::loadAll()
{
    return {};
}

std::vector<cl::Kernel> Compiler::build(const char* kernelName, CompileOption const& essentialFlag,
                                        CompileOption const& otherFlags) const
{
    cl::Program program{ context, GetSource(std::string{kernelName}+".cl") };
    try {
        program.build((essentialFlag.option+otherFlags.option).c_str());
    }
    catch (cl::Error& err) {
        std::cerr << "Build program error with flag" << essentialFlag.option << otherFlags.option << " Code: " << err.err() << '\n';
    }
    try {
        program.build(essentialFlag.option.c_str());
    }
    catch (cl::Error& err)
    {
        std::cerr << "Build program failed. Code: " << err.err() << '\n';
        const auto buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for (const auto& log : buildLog)
            std::cerr << log.first.getInfo<CL_DEVICE_NAME>() << ":\t" << log.second << '\n';
        throw std::runtime_error{ "Build program failed" };
    }
    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);
#ifdef DEBUG
    std::cerr << "Kernel creation success!\n";
#endif
    return kernels;
}

#ifndef ANDROID
std::vector<cl::Kernel> Compiler::build(std::filesystem::path const& path, CompileOption const& essentialFlag, CompileOption const& otherFlags) const
{
    cl::Program program{ context, GetSource(path.string()) };
    try {
        program.build((essentialFlag.option + otherFlags.option).c_str());
    }
    catch (cl::Error& err) {
        std::cerr << "Build program error with flag" << essentialFlag.option << otherFlags.option << " Code: " << err.err() << '\n';
    }
    try {
        program.build(essentialFlag.option.c_str());
    }
    catch (cl::Error& err)
    {
        std::cerr << "Build program failed. Code: " << err.err() << '\n';
        const auto buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for (const auto& log : buildLog)
            std::cerr << log.first.getInfo<CL_DEVICE_NAME>() << ":\t" << log.second << '\n';
        throw std::runtime_error{ "Build program failed" };
    }
    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);
#ifdef DEBUG
    std::cerr << "Kernel creation success!\n";
#endif
    return kernels;
}
#endif

void Compiler::saveKernel(const char* fileName, cl::Kernel const& kernel)
{
    auto const program = kernel.getInfo<CL_KERNEL_PROGRAM>();
    auto const binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    cl::Program::Binaries binaries(std::accumulate(binSizes.cbegin(), binSizes.cend(), 0ull));

    program.getInfo(CL_PROGRAM_BINARIES, &binaries);
    std::ofstream f{ fileName, std::ios::binary };
    for(auto const& bin:binaries)
        f.write(reinterpret_cast<const char*>(bin.data()), bin.size());
}

#ifndef ANDROID
void Compiler::saveKernel(std::filesystem::path const& path, cl::Kernel const& kernel)
{
    //auto const program = kernel.getInfo<CL_KERNEL_PROGRAM>();
    //auto const binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    //cl::Program::Binaries binaries(std::accumulate(binSizes.cbegin(), binSizes.cend(), 0ull));

    //program.getInfo(CL_PROGRAM_BINARIES, &binaries);
    //std::ofstream f{ path, std::ios::binary };
    //for (auto const& bin : binaries)
    //    f.write(reinterpret_cast<const char*>(bin.data()), bin.size());
    saveKernel((path.string() + ".bin").c_str(), kernel);
}
#endif

std::vector<cl::Kernel> Compiler::loadKernel(const char* fileName, std::vector<cl::Device> const& devices)
{
    std::ifstream f{ fileName, std::ios::binary };
    cl::Program::Binaries bin{ {std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}} };
    cl::Program program{ context, devices, bin };
    program.build();
    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);
    return kernels;
}

#ifndef ANDROID
std::vector<cl::Kernel> Compiler::loadKernel(std::filesystem::path const& path, std::vector<cl::Device> const& devices)
{
    std::ifstream f{ path, std::ios::binary };
    cl::Program::Binaries bin{ {std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}} };
    cl::Program program{ context, devices, bin };
    program.build();
    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);
    return kernels;
}
#endif
