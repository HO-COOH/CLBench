kernel void TestRead(global char* restrict ptr) 
{
    int id = get_global_id(0);
    ptr[id] = (id % 128);
}