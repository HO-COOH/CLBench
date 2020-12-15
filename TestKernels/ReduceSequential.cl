kernel void ReduceSequential(global float const *restrict src, global float *restrict dst, local float* groupData) 
{
    /*copy data -> local shared data*/
    unsigned int groupId = get_local_id(0);
    unsigned int globalId = get_global_id(0);
    groupData[groupId] = src[globalId];
    barrier(CLK_LOCAL_MEM_FENCE);

    /*do reduction in local shared data*/
    int const localSize=get_local_size(0);
    for(unsigned int i = localSize/2; i>0 ; i>>=1)
    {
        if(groupId < i)
            groupData[groupId] += groupData[groupId + i];
    }
    if(groupId == 0)
        dst[get_group_id(0)] = groupData[0];
}