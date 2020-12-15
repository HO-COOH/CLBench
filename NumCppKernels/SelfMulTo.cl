#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void SelfMulTo(global double* restrict result, global const double* restrict src, double value) 
{
    int const id=get_global_id(0);
    result[id]=src[id]*value;
}