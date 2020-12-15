typedef struct thing_t
{
    int data[3];
}Thing;

kernel void PassingStruct(Thing thing) 
{
    printf("Thing: %d, %d, %d\n", thing.data[0], thing.data[1], thing.data[2]);
}