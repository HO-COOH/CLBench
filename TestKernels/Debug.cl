kernel void Debug(global float const *restrict src, global float *restrict dst, local float* groupData) 
{
    /*copy data -> local shared data*/
    unsigned int groupId = get_local_id(0);
    unsigned int globalId = get_global_id(0);
    groupData[groupId] = src[globalId];
    barrier(CLK_LOCAL_MEM_FENCE);

    /*do reduction in local shared data*/
    for(unsigned int i = 1; i < get_local_size(0); i*=2)
    {
        if((groupId % (2 * i)) == 0)
            groupData[groupId] += groupData[groupId + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(groupId == 0)
        dst[get_group_id(0)] = groupData[0];
    printf("%f\n", src[get_global_id(0)]);
}