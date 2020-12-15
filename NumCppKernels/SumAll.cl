#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void SumAll(constant double *src, global double *restrict dst, local double* target) 
{
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int globalSize=get_global_size(0);
    const int localSize=get_local_size(0);
    const int groupId=get_group_id(0);
    const bool isLastGroup = ((globalSize/localSize) == groupId);
    /*Copy global -> local */
    target[lid] = src[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    /*Initial block size == work group size */
    size_t blockSize = (isLastGroup? globalSize%localSize : localSize);
    size_t halfBlockSize = blockSize / 2;
    while (halfBlockSize > 0) 
    {
        if(lid<halfBlockSize)   //half of the number of elements of thread doing reduce
        {
            target[lid]+=target[lid+halfBlockSize];
            if(halfBlockSize*2 < blockSize) //total number of elements is odd
            {
                if(lid==0)
                    target[0]+=target[blockSize-1]; //let the work_item[0] deal with the remaining element
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        blockSize=halfBlockSize;
        halfBlockSize=blockSize/2;
    }
    /*Write -> output */
    if(lid==0)
        dst[groupId]=target[0];
}