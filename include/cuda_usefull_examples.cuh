#include <cuda.h>
#include <stdio.h>

__host__ __device__ void testy()
{
  #ifdef __CUDA_ARCH__
  printf("GPU_called me\n");
  #else
  printf("CPU_called me\n");
  #endif
}

/*################################################################################
CUDA HELLO WORLD
################################################################################*/
__global__ void hello()
{
  if ( threadIdx.x ==0 and threadIdx.y==0)
  {
    testy();
  }
  printf("GPU: Hello World %d,%d block %d,%d\n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
}

/*################################################################################
CUDA HELLO WORLD
################################################################################*/
void launch_hello() 
{
  printf("CPU will call this function:\n");
  testy();
  printf("Not called from the GPU\n");
  dim3 threadsPerBlock(1, 1);
  dim3 numBlocks( 1, 1);
  checkCUDAErrorWithLine("Numblock did not work!");
  hello<<<numBlocks, threadsPerBlock>>>();
  checkCUDAErrorWithLine("Command not executed");
  cudaDeviceSynchronize();
  return;
}
