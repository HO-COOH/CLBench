#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void AddTwoTo(global double* restrict result, global const double* restrict lhs, global const double* restrict rhs) 
{
    int const id=get_global_id(0);
    result[id]=lhs[id]+rhs[id];
}