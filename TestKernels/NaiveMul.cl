size_t getIndex(int row, int col, int width) { return row*width+col; }

kernel void NaiveMul(global float* a, global float* b, global float* result) 
{
    int const row=get_global_id(0);
    int const col=get_global_id(1);
    int const a_cols=get_global_size(0);  //A should have the same number of columns as B's number of row

    global float* a_row=&a[getIndex(row, 0, a_cols)];//A[row][0]
    global float* b_col=&b[getIndex(0, col, a_cols)];//B[0][col]
    global float* const out=&result[getIndex(row, col, a_cols)];//Result[row][col]

    float sum=0.0f;
    for(int i=0; i<a_cols; ++i)
        sum+=a_row[i]*b_col[i];
    *out=sum;
}