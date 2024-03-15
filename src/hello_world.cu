#include <cuda.h>
#include <stdio.h>

__global__ void hello()
{
        printf("Hello World\n");
}

int main()
{
        int a = 0;
        hello<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
}
