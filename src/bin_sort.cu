#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "BOID_struct.h"

//################################################################################
// define variables
#define n_bins 10
#define n_boids 1000000
//################################################################################

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
/*################################################################################
Check for CUDA errors; print and exit if there was a problem.
################################################################################*/
void checkCUDAError(const char *msg, int line = -1)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/*################################################################################
* INFORMATION aboutthe GPU architecture
* for debugging purpuses only!
################################################################################*/
void get_GPU_information()
{
  int device_id;
  cudaDeviceProp prop;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&prop, device_id);
  
  int num_sm = prop.multiProcessorCount;
  int cache = prop.l2CacheSize;
  int num_cores_per_sm = prop.maxThreadsPerMultiProcessor;
  int cache_share_sm = prop.sharedMemPerMultiprocessor ;
  int cache_share_block = prop.sharedMemPerBlock ;
  int CUDA_cores = _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount;
  
  printf("######################################## INFO ########################################\n");
  printf("Number of SMs: %d\n",num_sm);
  printf("Number of  max cores per SM: %d\n", num_cores_per_sm );
  printf("Total number of CUDA cores: %d\n", CUDA_cores );
  printf("sharedMemPerMultiprocessor: %d\n",cache_share_sm);
  printf("sharedMemPerBlock: %d\n", cache_share_block);
  printf("l2CacheSize: %d\n", cache);
  printf("######################################## END  ########################################\n");  
  return;
}

/*################################################################################
* SPATIAL HASH FUNCTION
* implemented similar to: 
*   Ihmsen, M. et al. A Parallel SPH Implementation on Multi-Core CPUs. Computer Graphics Forum, 30: 99-112. 
*   https://doi.org/10.1111/j.1467-8659.2010.01832.x
################################################################################*/
__device__ int spatial_hash_3d(float x,float y,float z, float gridsize, int hashlist_length)
{
  int c = (
    (static_cast<int>(x/gridsize)*static_cast<int>(73856093)) ^
    (static_cast<int>(y/gridsize)*static_cast<int>(19349663)) ^
    (static_cast<int>(z/gridsize)*static_cast<int>(83492791))
  )%hashlist_length;
  return c;
}
__device__ int spatial_hash_2d(float x,float y, float gridsize, int hashlist_length)
{
  int c = static_cast<int>((
    (static_cast<int>(x/gridsize)*static_cast<int>(73856093)) ^
    (static_cast<int>(y/gridsize)*static_cast<int>(19349663))
    )%hashlist_length);
  return c;
}

/*################################################################################
* BIN SORTING
* implemented similar to: 
*   Gross, J. et al, Fast and Efficient Nearest Neighbor Search for Particle Simulations
*   https://doi.org/10.2312/cgvc.20191258
################################################################################*/
__global__ void bin_sort(Boid* d_boids, const int max_numberofboids , unsigned int* global_bins) 
{
  // initialise the shared bins for the local SM and set start value to 0
  // shared memor amount is speficied as the 3 argument in the kernel function call
  extern __shared__ int shared_bins[];
  for (int i = threadIdx.x; i < n_bins; i += blockDim.x) 
  {
    shared_bins[i] = 0;
  }
  __syncthreads();

  // get current ID of the thread (get current positition in the SM grid)
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // check for memory overflow, we just want to acces all boids and not outside.
  if (id < max_numberofboids)
  {
    // calculate HASH and increment appropriate bin
    int HASH = spatial_hash_3d(d_boids[id].position_x,d_boids[id].position_y,d_boids[id].position_z,static_cast<float>(5),n_bins);
    atomicAdd(&shared_bins[HASH], 1);
    
    // Print out the first 10 boids for debugging
    if (id <10)
    {
      printf("Boid %d - Position: (%f, %f, %f)__HASH: %d\n", id, d_boids[id].position_x, d_boids[id].position_y, d_boids[id].position_z, HASH);
    }
  }

  // Synchronize to ensure all threads have finished pushing to shared memory
  __syncthreads();

  // Copy the shared memory to global memory
  for (int i = threadIdx.x; i < n_bins; i += blockDim.x) 
  {
    atomicAdd(&global_bins[i], shared_bins[i]);
  }
  
  return;
}

void bin_sort_test()
{

  //print out some stats
  printf("number of boids: %d , number of bins: %d\n",n_boids,n_bins);

  // create array of boids
  Boid* boids = new Boid[n_boids];

  // just some predefined boids i can calculate by hand. Basically random numbers
  const float array_x[10] = {9.7, 5.0, 4.6, 1.1, 9.2, 3.0, 4.0, 9.9, 4.5, 6.7};
  const float array_y[10] = {5.7, 0.3, 2.7, 0.2, 8.5, 7.0, 7.6, 6.2, 8.9, 5.3};

  for (int i = 0; i<10 ; ++i)
  {
    boids[i].position_x = array_x[i];
    boids[i].position_y = array_y[i];
    boids[i].position_z = 0.;
  }
  for (int i = 10; i<n_boids ; ++i)
  {
    boids[i].position_x = 0.;
    boids[i].position_y = 0.;
    boids[i].position_z = 0.;
  }

  int not_used_but_never_mentioned = 42;

  // Allocate memory on device
  Boid* d_boids;
  cudaMalloc(&d_boids, n_boids * sizeof(Boid));
  checkCUDAErrorWithLine("Allocation d_boids");
  unsigned * d_bins;
  cudaMalloc(&d_bins, n_bins * sizeof(int));
  checkCUDAErrorWithLine("Allocation d_global_bins");

  // Copy Boid data from host to device
  cudaMemcpy(d_boids, boids, n_boids * sizeof(Boid), cudaMemcpyHostToDevice);
  checkCUDAErrorWithLine("Memory copy: BOIDS");

  // Copy host bins to device just to make sure, memomry is clear.
  int* h_bins = new int[n_bins];
  for (int i = 0; i<n_bins ; ++i)
  {
    h_bins[i] = 0;
  }
  cudaMemcpy(d_bins, h_bins, n_bins * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAErrorWithLine("Memory copy: BINS");

  // get the number of threads per block
  // cudaDeviceProp prop;
  // int device_id // = 0; change the device ID for used GPU
  // cudaGetDevice(&device_id); 
  // cudaGetDeviceProperties(&prop, device_id);
  // const int threads = prop.maxThreadsPerBlock/1;
  // const int threads = _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount;
  const int threads = 896;
  
  // calculate the number of thread and block to be used
  dim3 threadsPerBlock(threads );
  dim3 numBlocks( (int) ceil(n_boids/threads+1) );
  printf("thread: %d\nblocks: %d\n",threads,(int) ceil(n_boids/threads+1));

  // Launch kernel to print Boid data on the device
  auto start = std::chrono::steady_clock::now();
  // start cuda kernel and allocate shared bins memory dynamically.
  bin_sort<<<numBlocks, threadsPerBlock , n_bins*sizeof(int)>>>(d_boids,n_boids,d_bins);
  checkCUDAErrorWithLine("Kernel start");
  
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  std::cout << std::chrono::duration<double, std::milli>(diff).count() << " ms | ";
  std::cout << std::chrono::duration<double, std::micro>(diff).count() << " µs | ";
  std::cout << 1000./std::chrono::duration<double, std::milli>(diff).count() << " it/s" << std::endl;

  // get the bins from the device
  cudaMemcpy(h_bins, d_bins, n_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("d_bins: copy do host");

  // print bins
  // /*
  int count_in_bins = 0;
  printf("bins: %d\n",n_bins);
  for (int i = 0; i<n_bins ; ++i)
  {
    count_in_bins += h_bins[i];
    if (i<n_bins-1)
      if (h_bins[i]==0)
        printf("_ ");
      else      
        printf("%u  ",h_bins[i]);
    else
    {
      if (h_bins[i]==0)
        printf("_ \n");
      else
        printf("%u\n",h_bins[i]);
    }
  }
  printf("count in all bins: %d \n",count_in_bins);
  // */
    
  // Free allocated memory
  cudaFree(d_boids);
  checkCUDAErrorWithLine("d_boids: Free memory");
  cudaFree(d_bins);
  checkCUDAErrorWithLine("d_bins: Free memory");

  delete[] boids;
  delete[] h_bins;

  return;
}