__kernel void prime(__global bool* result, unsigned long start) 
{
    int id=get_global_id(0);
    unsigned long num=start+2 * id;
    bool isPrime=true;
    for(unsigned long i=3; i<num/2; ++i)
    {
        if(num%i==0)
        {
            isPrime=false;
            break;
        }
    }
    result[id]=isPrime;
}