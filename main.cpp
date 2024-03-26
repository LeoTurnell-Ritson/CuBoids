#include <cstdio>
#include <stdio.h>

#include "bin_sort.cuh"

#include <iostream>

int main()
{
  printf("################################################################################\n");
  printf("%d\n", static_cast<int>(static_cast<float>(19)/static_cast<float>(10)+1));

  get_GPU_information();

  bin_sort_test();
  
  return 0;
}
