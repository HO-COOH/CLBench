cmake_minimum_required(VERSION 3.16)

project(CLBench)

function(DownloadOpenCLHeaders)
    include(FetchContent)
    message(STATUS "Downloading OpenCL C headers:")
    #The opencl C headers
    FetchContent_Declare(
        OpenCLCHeader
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
        GIT_TAG origin/master
    )
    FetchContent_MakeAvailable(OpenCLCHeader)
    FetchContent_GetProperties(OpenCLCHeader SOURCE_DIR OpenCLCHeaderDir)
    include_directories(${OpenCLCHeaderDir})
    message(STATUS "Finished downloading OpenCL C headers!")

    #The opencl C++ headers
    message(STATUS "Downloading OpenCL C++ headers:")
    FetchContent_Declare(
        OpenCLCPPHeader
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP
        GIT_TAG origin/master
    )
    FetchContent_Populate(OpenCLCPPHeader)
    FetchContent_GetProperties(OpenCLCPPHeader SOURCE_DIR OpenCLCPPHeaderDir)
    include_directories(${OpenCLCPPHeaderDir}/include)
    message(STATUS "Finished downloading OpenCL C++ headers!")
endfunction()



include_directories(include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Main ./source/main.cpp 
    ./source/Compiler.cpp
    ./source/Error.cpp
    ./source/GPU.cpp
    ./source/KernelInfo.cpp
    ./source/Test.cpp
)
add_compile_definitions(CL_HPP_ENABLE_EXCEPTIONS)
set_target_properties(Main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Might want to change to use a higher opencl version
add_compile_definitions(CL_HPP_MINIMUM_OPENCL_VERSION=120)
add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=120)

if(NOT ANDROID)
    if(MSVC)
        add_compile_definitions(WIN32)
    endif()
    if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
        message("Debug build")
        add_compile_definitions(DEBUG)
    else()
        message("Release build")
    endif()

    if(WIN32) #For windows, use the dependencies
        DownloadOpenCLHeaders()
        add_library(OpenCL SHARED IMPORTED)
        find_file(OpenCLLibLocation OpenCL.dll PATHS "./dependency/" REQUIRED NO_DEFAULT_PATH)
        find_file(OpenCLImplLibLocation OpenCL.lib PATHS "./dependency/" REQUIRED NO_DEFAULT_PATH)
        set_target_properties(OpenCL PROPERTIES IMPORTED_LOCATION ${OpenCLLibLocation} IMPORTED_IMPLIB ${OpenCLImplLibLocation})
        target_link_libraries(Main PRIVATE OpenCL)
    else() #otherwise use vcpkg to get OpenCL
        find_package(OpenCL REQUIRED)
        target_link_libraries(Main PRIVATE OpenCL::OpenCL)
    endif()
else()

    DownloadOpenCLHeaders()
    #Find and kink with OpenCL lib
    add_library(OpenCL SHARED IMPORTED)
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        find_file(OpenCLLibLocation libOpenCL.so PATHS "/system/vendor/lib64" "/system/lib64" REQUIRED)
    else()
        find_file(OpenCLLibLocation libOpenCL.so PATHS "/system/vendor/lib" "/system/lib" REQUIRED)
    endif()
    add_compile_definitions(ANDROID)
    set_target_properties(OpenCL PROPERTIES IMPORTED_LOCATION ${OpenCLLibLocation})
    target_link_libraries(Main PRIVATE OpenCL)
endif()


# copy test files
set(CLNumCppCoreKernels
    "AddTwo"
    "AddTwoTo"
    "DivideTwo"
    "DivideTwoTo"
    "MinusTwo"
    "MinusTwoTo"
    "MulTwo"
    "MulTwoTo"
    "SelfAdd"
    "SelfAddTo"
    "SelfDivide"
    "SelfDivideTo"
    "SelfMinus"
    "SelfMinusTo"
    "SelfMul"
    "SelfMulTo"
    "SumAll"
)

#change the destination -> ${CMAKE_BINARY_DIR}/${BuildType} for MSVC
if(MSVC)
    if(NOT CMAKE_BUILD_TYPE)
        set(KernelDest ${CMAKE_BINARY_DIR}/Debug)
    else()
        set(KernelDest ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    endif()
endif()

if(NOT ANDROID)
    foreach(CLNumCppCoreKernel ${CLNumCppCoreKernels})
        message("Copying ${CLNumCppCoreKernel} -> build")
        message(${CLNumCppCoreKernel})
        file(
            COPY ${CMAKE_CURRENT_SOURCE_DIR}/NumCppKernels/${CLNumCppCoreKernel}.cl
            DESTINATION ${KernelDest}/test
        )
        file(
            COPY ${CMAKE_CURRENT_SOURCE_DIR}/NumCppKernels/${CLNumCppCoreKernel}.cl
            DESTINATION ${CMAKE_BINARY_DIR} #I tested visual studio and vscode using msvc will output to different directory, shitty solution
        )
    endforeach()
else()
    foreach(CLNumCppCoreKernel ${CLNumCppCoreKernels})
        message("Copying ${CLNumCppCoreKernel} -> build")
        message(${CLNumCppCoreKernel})
        file(
            COPY ${CMAKE_CURRENT_SOURCE_DIR}/NumCppKernelsAndroid/${CLNumCppCoreKernel}.cl
            DESTINATION ${KernelDest}/test
        )
    endforeach()
endif()
message(STATUS "Finish copying core kernels!")

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/TestKernels/*.cl")
foreach(file ${files})
    message("Copying ${file} -> build")
    file(
        COPY ${file}
        DESTINATION ${KernelDest}
    )
    file(
        COPY ${file}
        DESTINATION ${CMAKE_BINARY_DIR} #Same as above
    )
endforeach()
message(STATUS "Finish copying test kernels!")