#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void SelfAdd(__global double* data, double value) 
{
    data[get_global_id(0)]+=value;
}