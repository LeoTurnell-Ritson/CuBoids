#include <cstdio>
#include <stdio.h>

#include "bin_sort.cuh"

#include <iostream>

int main()
{
  printf("################################################################################\n");
  
  get_GPU_information();
  
  bin_sort_test();
  
  return 0;
}
