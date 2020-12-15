#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void SelfDevide(__global double* data, double value) 
{
    data[get_global_id(0)]/=value;
}