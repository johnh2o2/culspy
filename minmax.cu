
#ifndef _MINMAX_
#define _MINMAX_
#include<float.h>

float cpu_maxf(float *x, int n){
  int i;
  float m = -FLT_MIN;
  for(i=0; i<n; i++) if (x[i] > m) m = x[i];

  return m;
}

__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}



__global__ void max_reduce( float* d_array, 
                                    float *d_max,
                                    const size_t elements)
{
    //if (threadIdx.x == 0) d_max[blockIdx.x] = 999.;
    
    extern __shared__ float block_max[];
    //__shared__ float block_max[blockDim.x];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    block_max[tid] = -FLT_MAX; 
    //if (threadIdx.x == 0) d_max[blockIdx.x] = 999;
    
    while (gid < elements) {
        block_max[tid] = max(block_max[tid], d_array[gid]);
        gid += gridDim.x*blockDim.x;
        }
    __syncthreads();
    //if (threadIdx.x == 0) d_max[blockIdx.x] = 999;
    
    gid = (blockDim.x * blockIdx.x) + tid;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
            block_max[tid] = max(block_max[tid], block_max[tid + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0) d_max[blockIdx.x] = block_max[0];
    
}

#endif