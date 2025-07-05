#include "helpers.cuh"
#include "mylib.h"
// #include "cub_algorithms.cuh"
#include "stdio.h"
int main(int argc, char **argv)
{
  int res = mylib::add(3, 4);
  printf("test_mylib %d\n", res);

  int N = 50;
  float array[N];
  for (int i = 0; i < N; ++i)
  {
    array[i] = i + 0.0f;
  }
  float ret = 0; // infini_cub::cub_sum_wrapper(array, N);
  printf("ret11: %f\n", ret);
}
