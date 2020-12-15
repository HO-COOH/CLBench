size_t getIndex(int row, int col, int width) { return row*width+col; }

__kernel void BlockMul(__global const float* a, __global const float* b, __local float* a_local, __local float* b_local, __global float* result) 
{
    int const row=get_global_id(0);
    int const col=get_global_id(1);

    int const limit=get_global_size(0);

    int const blockSize=get_local_size(0);
    int const gidRow=get_local_id(0);
    int const gidCol=get_local_id(1);

    float sum=0.0f;
    for(int blockIndex=0; blockIndex < (limit/blockSize); ++blockIndex)
    {
        /*copy -> local memory*/
        a_local[getIndex(gidRow, gidCol, blockSize)]=a[getIndex(row, col, limit)];
        b_local[getIndex(gidRow, gidCol, blockSize)]=b[getIndex(row, col, limit)];
        barrier(CLK_LOCAL_MEM_FENCE);

        /*block multiply*/
        local float* restrict a_local_row=&a_local[getIndex(gidRow, 0, blockSize)];
        local float* restrict b_local_row=&b_local[getIndex(gidRow, 0, blockSize)];
        for(int i=0; i<blockSize; ++i)
        {
            sum+= (*a_local_row) * (*b_local_row);
            ++a_local_row;
            ++b_local_row;
        }
    }

    result[getIndex(row, col, limit)]=sum;
}