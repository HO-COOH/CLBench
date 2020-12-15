
kernel void AddTwo(global float* lhs, global float* rhs) 
{
    int const id=get_global_id(0);
    lhs[id]/=rhs[id];
}