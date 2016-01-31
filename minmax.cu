
#ifndef _MINMAX_
#define _MINMAX_

#include<float.h>

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

__global__ void init_dinds( int *d_inds, int N){
    int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
    d_inds[gid] = gid;   
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

__global__ void max_reduce_with_index( float* d_array, int *d_inds, 
                                    float *d_max, int *d_imax,
                                    const size_t elements)
{
    //if (threadIdx.x == 0) d_max[blockIdx.x] = 999.;
    
    __shared__ float block_max[BLOCK_SIZE];
    __shared__ int block_imax[BLOCK_SIZE];
    //__shared__ float block_max[blockDim.x];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    block_max[tid] = d_array[0];
    block_imax[tid] = d_inds[0];

    //if (threadIdx.x == 0) d_max[blockIdx.x] = 999;
    
    while (gid < elements) {
        if (d_array[gid]> block_max[tid]){
            block_imax[tid] = d_inds[gid];
            block_max[tid] = d_array[gid];
        }
        gid += gridDim.x*blockDim.x;
    }
    __syncthreads();
    //if (threadIdx.x == 0) d_max[blockIdx.x] = 999;
    
    gid = (blockDim.x * blockIdx.x) + tid;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements){
            if (block_max[tid + s] > block_max[tid]){
                block_imax[tid] = block_imax[tid + s];
                block_max[tid] = block_max[tid + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_max[blockIdx.x] = block_max[0];
        d_imax[blockIdx.x] = block_imax[0];
    }
    
}



float cpu_maxf(float *x, int n){
  int i;
  float m = -FLT_MIN;
  for(i=0; i<n; i++) if (x[i] > m) m = x[i];

  return m;
}

void
cpu_maxf_ind(float *x, int N, float *m, int *ind){
  int i;
  *m = x[0];
  *ind = 0;
  for(i=0; i< N; i++){
    if (x[i] > *m) {
      *m = x[i];
      *ind = i;
    }
  }
}

void 
cpu_stats(float *x, int N, float *mu, float *std){
  int i;
  *mu = 0; *std = 0;
  for (i=0; i<N; i++){
    *mu += x[i];
    printf(" + %.3e = %.3f (%d) - ", x[i], *mu, i);
  }
  *mu /= N;
  printf("FINAL %.3f\n", *mu);
  exit(EXIT_FAILURE);

  for(i=0; i<N; i++){
    *std += (x[i] - *mu) * (x[i] - *mu);
  }
  *std = sqrt(*std/N);
}


void 
gpu_maxf_ind(float *d_X, int N, float *m, int *ind){
  
  int gd = N/BLOCK_SIZE, gdm, gdm0;
  float *d_maxbuf;
  int *d_inds, *d_imaxbuf;


  while (gd * BLOCK_SIZE < N) gd += 1;
  dim3 grid_dim(gd, 1, 1);
  dim3 block_dim(BLOCK_SIZE, 1, 1);
    
  cudaMalloc((void **) &d_imaxbuf, gd * sizeof(int));
  cudaMalloc((void **) &d_maxbuf, gd * sizeof(float));
  cudaMalloc((void **) &d_inds, N * sizeof(int));

  // calculate the maximum.
  init_dinds<<<grid_dim, block_dim>>>(d_inds, gd);
  max_reduce_with_index<<<grid_dim, block_dim>>>(d_X, d_inds, d_maxbuf, d_imaxbuf, N);
  
  // Now reduce until only one block is needed.
  gdm = gd;
  while (gdm > 1){

    gdm0 = gdm;
    gdm /= BLOCK_SIZE;
    if( gdm * BLOCK_SIZE < gdm0 ) gdm += 1;
    
    dim3 grid_dim_max(gdm, 1, 1);

    max_reduce_with_index<<<grid_dim_max, block_dim>>>(d_maxbuf, d_imaxbuf, 
                                      d_maxbuf, d_imaxbuf, gdm0);
  
  }

  //copy max(P) to the host
  cudaMemcpy(m, d_maxbuf, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(ind, d_imaxbuf, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_imaxbuf);
  cudaFree(d_maxbuf);
  cudaFree(d_inds);

}

#endif