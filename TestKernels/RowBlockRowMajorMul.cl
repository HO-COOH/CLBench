size_t getIndex(int row, int col, int width) 
{ 
    return row*width+col; 
}
#define blockSize 8
kernel void RowBlockRowMajorMul(
    global const float* restrict a, 
    global const float* restrict b,
    local float* restrict a_local,
    local float* restrict b_local, 
    global float* restrict result) 
{
    int const row=get_global_id(0);
    int const col=get_global_id(1);

    int const groupRow=get_group_id(0);
    int const groupCol=get_group_id(1);
    int const flattenedGroupId=groupRow*get_num_groups(1)+groupCol;


    int const limit=get_global_size(0);

    int const blockCount=blockSize*blockSize;

    int const gidRow=get_local_id(0);
    int const gidCol=get_local_id(1);
    int const flattenedGid=gidRow*blockSize+gidCol;

    float sum=0.0f;
    int offset=flattenedGroupId*blockSize*blockSize+flattenedGid;
    for(int blockIndex=0; blockIndex < (limit/blockSize); ++blockIndex)
    {
        /*copy -> local memory*/
        a_local[flattenedGid]=a[offset];
        b_local[flattenedGid]=b[offset];
        barrier(CLK_LOCAL_MEM_FENCE);

        /*block multiply*/
        for(int i=0; i<blockSize; ++i)
        {
            //sum+= a_local[getIndex(gidRow, i, blockSize)]*b_local[getIndex(gidRow, i, blockSize)];
            sum+= a_local[getIndex(gidRow, i, blockSize)]*b_local[getIndex(gidRow, i, blockSize)];
        }
    }
    result[getIndex(row, col, limit)]=sum;
}