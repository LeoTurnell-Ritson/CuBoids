#include <cuda.h>

__global__ void hello()
{
        printf("Hello World\n");
}

int main()
{
        hello<<<1, 1>>>();
        return 0;
}
