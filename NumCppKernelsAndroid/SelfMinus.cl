
kernel void SelfMinus(__global float* data, float value) 
{
    data[get_global_id(0)]-=value;
}