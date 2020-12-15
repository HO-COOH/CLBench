/*****************************************************************//**
 * \file   Error.hpp
 * \brief  Core error library
 * 
 * \author Peter
 * \date   September 2020
 *********************************************************************/
#pragma once


#include <CL/opencl.hpp>
class ValueError:public std::exception
{
    
};

class NotImplementException:public std::exception
{
    
};

class InvalidAccessMode: public std::exception
{
    
};

/**
 * @brief Get OpenCL error message corresponded to the error code
 * @param err OpenCL error code
 * @return The corresponding error message
 */
const char* GetErrorDescription(cl_int err);

/**
 * @brief Print OpenCL error message with a description
 * @param err Reference to the OpenCL exception object
 */
void PrintCLError(cl::Error const& err);
