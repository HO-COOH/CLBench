kernel void test(global const char* ptr, unsigned long size) 
{
    printf("%d, %d", ptr[0], ptr[size]);
}