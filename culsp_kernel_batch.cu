// Copyright 2010 Rich Townsend <townsend@astro.wisc.edu>
//
// This file is part of CULSP.
//
// CULSP is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CULSP is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CULSP. If not, see <http://www.gnu.org/licenses/>.

#ifndef _CULSP_KERNEL_BATCH_
#define _CULSP_KERNEL_BATCH_



__global__ void
__launch_bounds__(BLOCK_SIZE)
dummy_kernel(float *d_t, float *d_X, float *d_P, float df, 
                      int offset, int N_t, int N_f, float minf)
{
  /*
  int id = blockIdx.x*BLOCK_SIZE+threadIdx.x;
 
  int i;

  if ( id >= N_f ) return;

  __shared__ float s_t[BLOCK_SIZE];
  __shared__ float s_X[BLOCK_SIZE];

  // Calculate the frequency

  float f = (id+1)*df + minf;

  // Calculate the various sums

  float XC = 0.f;
  float XS = 0.f;
  float CC = 0.f;
  float CS = 0.f;

  float XC_chunk = 0.f;
  float XS_chunk = 0.f;
  float CC_chunk = 0.f;
  float CS_chunk = 0.f;

  int j;
  */
  return;
}

__global__ void
__launch_bounds__(BLOCK_SIZE)
culsp_kernel_stream(float *d_t, float *d_X, float *d_P, float df, 
                      int offset, int N_t, int N_f, float minf)
{
  int id = blockIdx.x*BLOCK_SIZE+threadIdx.x;
 
  int i;

  if ( id >= N_f ) return;

  __shared__ float s_t[BLOCK_SIZE];
  __shared__ float s_X[BLOCK_SIZE];

  // Calculate the frequency

  float f = (id+1)*df + minf;

  // Calculate the various sums

  float XC = 0.f;
  float XS = 0.f;
  float CC = 0.f;
  float CS = 0.f;

  float XC_chunk = 0.f;
  float XS_chunk = 0.f;
  float CC_chunk = 0.f;
  float CS_chunk = 0.f;

  int j;

  for(j = 0; j < N_t- BLOCK_SIZE; j += BLOCK_SIZE) {

    // Load the chunk into shared memory

    __syncthreads();

    s_t[threadIdx.x] = d_t[offset + j+threadIdx.x];
    s_X[threadIdx.x] = d_X[offset + j+threadIdx.x];

    __syncthreads();

    // Update the sums

    #pragma unroll
    for(int k = 0; k < BLOCK_SIZE; k++) {

      // Range reduction

      float ft = f*s_t[k];
      ft -= rintf(ft);

      float c;
      float s;

      __sincosf(TWOPI*ft, &s, &c);

      XC_chunk += s_X[k]*c;
      XS_chunk += s_X[k]*s;
      CC_chunk += c*c;
      CS_chunk += c*s;

    }

    XC += XC_chunk;
    XS += XS_chunk;
    CC += CC_chunk;
    CS += CS_chunk;

    XC_chunk = 0.f;
    XS_chunk = 0.f;
    CC_chunk = 0.f;
    CS_chunk = 0.f;
    
  }

  // Handle the final chunk

  __syncthreads();

  if(j+threadIdx.x < N_t) {

    s_t[threadIdx.x] = d_t[offset + j+threadIdx.x];
    s_X[threadIdx.x] = d_X[offset + j+threadIdx.x];

  }

  __syncthreads();
    
  for(int k = 0; k < N_t-j; k++) {

    // Range reduction

    float ft = f*s_t[k];
    ft -= rintf(ft);

    float c;
    float s;

    __sincosf(TWOPI*ft, &s, &c);

    XC_chunk += s_X[k]*c;
    XS_chunk += s_X[k]*s;
    CC_chunk += c*c;
    CS_chunk += c*s;

  }

  XC += XC_chunk;
  XS += XS_chunk;
  CC += CC_chunk;
  CS += CS_chunk;

  float SS = (float) N_t - CC;
    
  // Calculate the tau terms

  float ct;
  float st;

  __sincosf(0.5f*atan2(2.f*CS, CC-SS), &st, &ct);

  // Calculate P

  d_P[id] = 
      0.5f*((ct*XC + st*XS)*(ct*XC + st*XS)/
      (ct*ct*CC + 2*ct*st*CS + st*st*SS) + 
      (ct*XS - st*XC)*(ct*XS - st*XC)/
      (ct*ct*SS - 2*ct*st*CS + st*st*CC));

  // Finish

}


__global__ void
__launch_bounds__(BLOCK_SIZE)
culsp_kernel_batch(float *d_t, float *d_X, float *d_P, float df, 
                      int *N_t, int Nlc, int N_f, float minf)
{
  int id = blockIdx.x*BLOCK_SIZE+threadIdx.x;
  int lcno = id % N_f;
  int offset = 0;
  int i;
  for(i=0; i<lcno; i++) offset += N_t[i];
 
  if ( id >= N_f*Nlc || lcno >= Nlc ) return;

  __shared__ float s_t[BLOCK_SIZE];
  __shared__ float s_X[BLOCK_SIZE];

  // Calculate the frequency

  float f = (id+1)*df + minf;

  // Calculate the various sums

  float XC = 0.f;
  float XS = 0.f;
  float CC = 0.f;
  float CS = 0.f;

  float XC_chunk = 0.f;
  float XS_chunk = 0.f;
  float CC_chunk = 0.f;
  float CS_chunk = 0.f;

  int j;

  for(j = 0; j < N_t[lcno]-BLOCK_SIZE; j += BLOCK_SIZE) {

    // Load the chunk into shared memory

    __syncthreads();

    s_t[threadIdx.x] = d_t[offset + j+threadIdx.x];
    s_X[threadIdx.x] = d_X[offset + j+threadIdx.x];

    __syncthreads();

    // Update the sums

    #pragma unroll
    for(int k = 0; k < BLOCK_SIZE; k++) {

      // Range reduction

      float ft = f*s_t[k];
      ft -= rintf(ft);

      float c;
      float s;

      __sincosf(TWOPI*ft, &s, &c);

      XC_chunk += s_X[k]*c;
      XS_chunk += s_X[k]*s;
      CC_chunk += c*c;
      CS_chunk += c*s;

    }

    XC += XC_chunk;
    XS += XS_chunk;
    CC += CC_chunk;
    CS += CS_chunk;

    XC_chunk = 0.f;
    XS_chunk = 0.f;
    CC_chunk = 0.f;
    CS_chunk = 0.f;
    
  }

  // Handle the final chunk

  __syncthreads();

  if(j+threadIdx.x < N_t[lcno]) {

    s_t[threadIdx.x] = d_t[offset + j+threadIdx.x];
    s_X[threadIdx.x] = d_X[offset + j+threadIdx.x];

  }

  __syncthreads();
    
  for(int k = 0; k < N_t[lcno]-j; k++) {

    // Range reduction

    float ft = f*s_t[k];
    ft -= rintf(ft);

    float c;
    float s;

    __sincosf(TWOPI*ft, &s, &c);

    XC_chunk += s_X[k]*c;
    XS_chunk += s_X[k]*s;
    CC_chunk += c*c;
    CS_chunk += c*s;

  }

  XC += XC_chunk;
  XS += XS_chunk;
  CC += CC_chunk;
  CS += CS_chunk;

  float SS = (float) N_t[lcno] - CC;
    
  // Calculate the tau terms

  float ct;
  float st;

  __sincosf(0.5f*atan2(2.f*CS, CC-SS), &st, &ct);

  // Calculate P

  d_P[id] = 
      0.5f*((ct*XC + st*XS)*(ct*XC + st*XS)/
	    (ct*ct*CC + 2*ct*st*CS + st*st*SS) + 
	    (ct*XS - st*XC)*(ct*XS - st*XC)/
	    (ct*ct*SS - 2*ct*st*CS + st*st*CC));

  // Finish

}


__global__ void
__launch_bounds__(BLOCK_SIZE)
bootstrap_kernel_batch(float *d_t, float *d_X, float *d_P, float df, 
                  int *N_t, int Nlc, int N_f, float minf, curandState *state){

  // Same as culsp kernel, except that we draw an s_X value at random.
  // doing this N times will give you a statistical bootstrap from which
  // false alarm probabilities can be calculated. This gets rid of the Gaussian
  // error assumption, but does NOT relax the assumption that all observations
  // are uncorrelated (violated when you have red noise, etc.)
  int id = blockIdx.x*BLOCK_SIZE+threadIdx.x;
  int lcno = id % N_f;
  int offset = 0;
  int i;
  for (i=0; i < lcno; i++) offset+=N_t[i];
  if ( id >= N_f*Nlc) return;
  
  

  __shared__ float s_t[BLOCK_SIZE];
  __shared__ float s_X[BLOCK_SIZE];

  // Calculate the frequency

  float f = (id+1)*df + minf;

  // Calculate the various sums

  float XC = 0.f;
  float XS = 0.f;
  float CC = 0.f;
  float CS = 0.f;

  float XC_chunk = 0.f;
  float XS_chunk = 0.f;
  float CC_chunk = 0.f;
  float CS_chunk = 0.f;

  int j, jrand;

  for(j = 0; j < N_t[lcno]-BLOCK_SIZE; j += BLOCK_SIZE) {

    // Load the chunk into shared memory

    __syncthreads();

    jrand = (int) (N_t[lcno] * curand_uniform(&state[id]) );

    s_t[threadIdx.x] = d_t[offset + j+threadIdx.x];
    s_X[threadIdx.x] = d_X[offset + jrand];

    __syncthreads();

    // Update the sums

    #pragma unroll
    for(int k = 0; k < BLOCK_SIZE; k++) {

      // Range reduction

      float ft = f*s_t[k];
      ft -= rintf(ft);

      float c;
      float s;

      __sincosf(TWOPI*ft, &s, &c);

      XC_chunk += s_X[k]*c;
      XS_chunk += s_X[k]*s;
      CC_chunk += c*c;
      CS_chunk += c*s;

    }

    XC += XC_chunk;
    XS += XS_chunk;
    CC += CC_chunk;
    CS += CS_chunk;

    XC_chunk = 0.f;
    XS_chunk = 0.f;
    CC_chunk = 0.f;
    CS_chunk = 0.f;
    
  }

  // Handle the final chunk

  __syncthreads();

  if(j+threadIdx.x < N_t[lcno]) {
    jrand = (int) (N_t[lcno] * curand_uniform(&state[id]));

    s_t[threadIdx.x] = d_t[offset + j+threadIdx.x];
    s_X[threadIdx.x] = d_X[offset + jrand];

  }

  __syncthreads();
    
  for(int k = 0; k < N_t[lcno]-j; k++) {

    // Range reduction

    float ft = f*s_t[k];
    ft -= rintf(ft);

    float c;
    float s;

    __sincosf(TWOPI*ft, &s, &c);

    XC_chunk += s_X[k]*c;
    XS_chunk += s_X[k]*s;
    CC_chunk += c*c;
    CS_chunk += c*s;

  }

  XC += XC_chunk;
  XS += XS_chunk;
  CC += CC_chunk;
  CS += CS_chunk;

  float SS = (float) N_t[lcno] - CC;
    
  // Calculate the tau terms

  float ct;
  float st;

  __sincosf(0.5f*atan2(2.f*CS, CC-SS), &st, &ct);

  // Calculate P

  d_P[id] = 
      0.5f*((ct*XC + st*XS)*(ct*XC + st*XS)/
      (ct*ct*CC + 2*ct*st*CS + st*st*SS) + 
      (ct*XS - st*XC)*(ct*XS - st*XC)/
      (ct*ct*SS - 2*ct*st*CS + st*st*CC));

  // Finish

}



#endif
