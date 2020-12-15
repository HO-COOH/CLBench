
kernel void MulTwoTo(global float* restrict result, global const float* restrict lhs, global const float* restrict rhs) 
{
    int const id=get_global_id(0);
    result[id]=lhs[id]*rhs[id];
}