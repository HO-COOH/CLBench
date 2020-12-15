#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void MulTwo(global double* lhs, global double* rhs) 
{
    int const id=get_global_id(0);
    lhs[id]*=rhs[id];
}