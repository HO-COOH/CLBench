size_t dstOffset(int row, int col, int width)
{
    return (row*width+col);
}


typedef struct filter3_t
{
    float data[9];
} Filter3;

#define HALF_FILTER_SIZE 1
size_t srcOffset3(int row, int col, int width) 
{ 
    return ((row+HALF_FILTER_SIZE)*(2*HALF_FILTER_SIZE+width)+(col+HALF_FILTER_SIZE)); 
}
kernel void UnrolledConv3(global unsigned char const* restrict input, global unsigned char* restrict output, Filter3 filter) 
{
    int const row=get_global_id(0);
    int const col=get_global_id(1);
    int const width=get_global_size(1);

    float sum=
        input[srcOffset3(row-1, col-1, width)]*filter.data[0] + input[srcOffset3(row-1, col, width)]*filter.data[1] + input[srcOffset3(row-1, col+1, width)]*filter.data[2] + 
        input[srcOffset3(row, col-1, width)]*filter.data[3] + input[srcOffset3(row, col, width)]*filter.data[4] + input[srcOffset3(row, col+1, width)]*filter.data[5] + 
        input[srcOffset3(row+1, col-1, width)]*filter.data[6] + input[srcOffset3(row+1, col, width)]*filter.data[7] + input[srcOffset3(row+1, col+1, width)]*filter.data[8];

    output[dstOffset(row, col, width)] = sum;
}


#undef HALF_FILTER_SIZE
#define HALF_FILTER_SIZE 2
typedef struct filter5_t
{
    float data[25];
} Filter5;

size_t srcOffset5(int row, int col, int width) 
{ 
    return ((row+HALF_FILTER_SIZE)*(2*HALF_FILTER_SIZE+width)+(col+HALF_FILTER_SIZE)); 
}
kernel void UnrolledConv5(global unsigned char const* restrict input, global unsigned char* restrict output, Filter5 filter) 
{
    int const row=get_global_id(0);
    int const col=get_global_id(1);
    int const width=get_global_size(1);

    float sum=
        input[srcOffset5(row-2, col-2, width)]*filter.data[0] + input[srcOffset5(row-2, col-1, width)]*filter.data[1] + input[srcOffset5(row-2, col, width)]*filter.data[2] + input[srcOffset5(row-2, col+1, width)]*filter.data[3] + input[srcOffset5(row-2, col+2, width)]*filter.data[4] +
        input[srcOffset5(row-1, col-2, width)]*filter.data[5] + input[srcOffset5(row-1, col-1, width)]*filter.data[6] + input[srcOffset5(row-1, col, width)]*filter.data[7] + input[srcOffset5(row-1, col+1, width)]*filter.data[8] + input[srcOffset5(row-1, col+2, width)]*filter.data[9] +
        input[srcOffset5(row, col-2, width)]*filter.data[10] + input[srcOffset5(row, col-1, width)]*filter.data[11] + input[srcOffset5(row, col, width)]*filter.data[12] + input[srcOffset5(row, col+1, width)]*filter.data[13] + input[srcOffset5(row, col+2, width)]*filter.data[14] +
        input[srcOffset5(row+1, col-2, width)]*filter.data[15] + input[srcOffset5(row+1, col-1, width)]*filter.data[16] + input[srcOffset5(row+1, col, width)]*filter.data[17] + input[srcOffset5(row+1, col+1, width)]*filter.data[18] + input[srcOffset5(row+1, col+2, width)]*filter.data[19] +
        input[srcOffset5(row+2, col-2, width)]*filter.data[20] + input[srcOffset5(row+2, col-1, width)]*filter.data[21] + input[srcOffset5(row+2, col, width)]*filter.data[22] + input[srcOffset5(row+2, col+1, width)]*filter.data[23] + input[srcOffset5(row+2, col+2, width)]*filter.data[24];

    output[dstOffset(row, col, width)] = sum;
}

