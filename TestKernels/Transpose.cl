size_t getIndex(int row, int col, int width) { return row*width+col; }

kernel void Transpose(global float const* restrict src, global float* restrict dst) 
{
    int const row=get_global_id(0);
    int const col=get_global_id(1);

    dst[getIndex(col, row, get_global_size(0))]=src[getIndex(row, col, get_global_size(1))];
}