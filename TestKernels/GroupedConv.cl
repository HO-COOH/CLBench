typedef struct filter_t
{
    float data[(2*HALF_FILTER_SIZE+1)*(2*HALF_FILTER_SIZE+1)*CHANNELS];
} Filter;

size_t srcOffset(int row, int col, int width, int channel) 
{ 
    return ((row+HALF_FILTER_SIZE)*(2*HALF_FILTER_SIZE+width)+(col+HALF_FILTER_SIZE))*CHANNELS + channel; 
}

size_t dstOffset(int row, int col, int width, int channel)
{
    return (row*width+col)*CHANNELS + channel;
}

size_t filterOffset(int row, int col, int channel)
{
    return (row*(2*HALF_FILTER_SIZE+1)+col)*CHANNELS + channel;
}

kernel void NaiveConv(global unsigned char const* restrict input, global unsigned char* restrict output, Filter filter, local float* localData) 
{
    int const row=get_global_id(0);
    int const col=get_global_id(1);
    int const width=get_global_size(1);

    float sumChannel[CHANNELS]={0.0f};

    int const rowLocal=get_local_id(0);
    int const colLocal=get_local_id(1);
    int const localRows=get_local_size(0);

    /*copy from global -> local */
    for(int i=0; i<CHANNELS; ++i)
        localData[dstOffset(rowLocal, colLocal, localRows ,i)]=input[srcOffset(row, col, width, i)];
    barrier(CLK_LOCAL_MEM_FENCE);


    for(int i=-HALF_FILTER_SIZE; i<=HALF_FILTER_SIZE; ++i)
    {
        for(int j=-HALF_FILTER_SIZE; j<=HALF_FILTER_SIZE; ++j)
        {
            for(int channel=0; channel<CHANNELS; ++channel)
            {
                //Need fix for bordering pixels
                sumChannel[channel]+=(localData[dstOffset(row+i, col+j, width, channel)]*filter.data[filterOffset(i+HALF_FILTER_SIZE, j+HALF_FILTER_SIZE, channel)]);
            }
        }
    }

    global unsigned char* out=&output[dstOffset(row, col, width, 0)];
    for(int channel=0; channel<CHANNELS; ++channel)
    {
        *out=sumChannel[channel];
        ++out;
    }
}