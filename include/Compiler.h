/*****************************************************************//**
 * \file   Compiler.h
 * \brief  OpenCL compiler related
 * 
 * \author Wenhao Li
 * \date   September 2020
 *********************************************************************/
#pragma once

#include <string>
#include <CL/opencl.hpp>
#include <unordered_map>


#ifndef ANDROID
    #include <filesystem>
#endif

class Compiler;
struct CompileOption
{
private:
    std::string option;
public:
    struct Flag
    {
        const char* flag;
        constexpr operator const char* () const
        {
            return flag;
        }
    };

    struct Macro
    {
        const char* name;
        const char* definition = nullptr;
        operator std::string() const
        {
            std::string temp{ "-D " };
            temp += name;
            if (definition)
                (temp += "=") += definition;
            return temp;
        }
    };

    struct Std
    {
        constexpr static Flag CL1_1{ "-cl-std=CL1.1" };
        constexpr static Flag CL1_2{ "-cl-std=CL1.2" };
        constexpr static Flag CL2_0{ "-cl-std=CL2.0" };
    };
    struct Optimize
    {
        constexpr static Flag None{ "-cl-opt-disable" };
        constexpr static Flag EnableMad{ "-cl-mad-enable" };
        constexpr static Flag NoSignedZero{ "-cl-no-signed-zeros" };
        constexpr static Flag Level3{ "-O3" };
        constexpr static Flag FiniteMath{ "-cl-finite-math-only" };
        constexpr static Flag UnsafeMath{ "-cl-unsafe-math-optimizations" };
        constexpr static Flag FastMath{ "-cl-fast-relaxed-math" };/*This option includes the -cl-no-signed-zeros and -cl-mad-enable options.*/   
        constexpr static Flag UniformWorkGroupSize{ "-cl-uniform-work-group-size" };
    };
    struct Warning
    {
        constexpr static Flag None{ "-w" };
        constexpr static Flag All{ "-Werror" };
    };

    template<typename ...Options>
    CompileOption(Options&&... options):option{" "}
    {
        (((option += options) += ' '), ...);
    }

    friend class Compiler;
};

class Compiler
{
    cl::Context const& context;
public:
    Compiler(cl::Context const& context):context{context}{}

    [[nodiscard]]std::vector<cl::Kernel> build(const char* kernelName) const;
    [[nodiscard]] std::vector<cl::Kernel> build(const char* kernelName, CompileOption const& essentialFlag, CompileOption const& otherFlags) const;
#ifndef ANDROID
    [[nodiscard]] std::vector<cl::Kernel> build(std::filesystem::path const& path, CompileOption const& essentialFlag, CompileOption const& otherFlags) const;
#endif
    [[nodiscard]]std::vector<cl::Kernel> build(const char* kernelName, CompileOption const& essentialFlag) const;
    void buildAll() const;

#ifndef ANDROID
    void buildAll(std::unordered_map<std::string, cl::Kernel>& storage, CompileOption const& essentialFlag, CompileOption const& otherFlags) const;
#endif
    void buildAll(CompileOption const& essentialFlag) const;

    /**
     * @brief Save the compiled kernel into a binary file
     * @param fileName the name of the file to save
     * @param kernel the compiled OpenCl kernel
     */
    static void saveKernel(const char* fileName, cl::Kernel const& kernel);

#ifndef ANDROID
    static void saveKernel(std::filesystem::path const& path, cl::Kernel const& kernel);
#endif

    /**
     * @brief Load kernel from binary file
     * @param fileName The name of the binary file
     * @param devices A vector of cl::Devices
     * @return 
     */
    std::vector<cl::Kernel> loadKernel(const char* fileName, std::vector<cl::Device> const& devices);

#ifndef ANDROID
    std::vector<cl::Kernel> loadKernel(std::filesystem::path const& path, std::vector<cl::Device> const& devices);
#endif

    static std::vector<cl::Kernel> loadAll();

};

