
kernel void SelfDivideTo(global float* restrict result, global const float* restrict src, float value) 
{
    int const id=get_global_id(0);
    result[id]=src[id]/value;
}