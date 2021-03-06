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

// Includes

#include <stdlib.h>
#include <string.h>
#include <argtable2.h>

#include "periodogram.h"
#include "culsp.h"
#include "culsp_kernel.cu"
#include "minmax.cu"

// Wrapper macros

#define CUDA_CALL(call) {						\
    cudaError err = call;						\
    if(err != cudaSuccess) {						\
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	      __FILE__, __LINE__, cudaGetErrorString(err));		\
      exit(EXIT_FAILURE);						\
    }}

#define CUDA_ERR_CHECK() { \
    err = cudaGetLastError(); \
  if(err != cudaSuccess) { \
    fprintf(stderr, "Cuda error: kernel launch failed in file '%s' in line %i : %s.\n", \
      __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
  }}


// Forward declarations

//void initialize (int, char **, char **, char **, float *, float *, int *);
//void initialize_cuda (int, int);
//void eval_LS_periodogram (int, int, float, float *, float *, float *);

// Main program

int
main( int argc, char** argv) 
{

  char *filename_in;
  char *filename_out;
  float F_over;
  float F_high;
  int device;

  int N_t;
  int N_f;
 
  float *t;
  float *X;

  float df;
  float *P;
  float minf = 0.0;

  // Initialize

  initialize(argc, argv, &filename_in, &filename_out, &F_over, &F_high, &device);

  // Read the light curve

  read_light_curve(filename_in, &N_t, &t, &X);

  // Set up the frequency parameters

  set_frequency_params(N_t, t, F_over, F_high, &N_f, &df);

  // Allocate space for the periodogram

  P = (float *) malloc(N_f*sizeof(float));

  // Initialize CUDA

  initialize_cuda(device);

  // Start the timer

  double time_a = get_time();
		 
  // Evaluate the Lomb-Scargle periodogram
  // set minf to 0 here (simplicity; I'm interacting with this
  // through python anyway.
  eval_LS_periodogram(N_t, N_f, df, minf,  t, X, P);

  // Stop the timer

  double time_b = get_time();
  printf( "Processing time: %16.3f (ms)\n", (time_b-time_a)*1000);

  // Write the data to file

  write_periodogram(filename_out, N_f, df, P);

  // Free up space

  free(P);

  free(X);
  free(t);

  // Finish
  return 0;

}


////
// Initialization
////

void
initialize (int argc, char **argv, char **filename_in, char **filename_out, 
	    float *F_over, float *F_high, int *device)
{

  // Set up the argtable structs

  struct arg_file *in = arg_file1(NULL, "in", "<filename_in>", "input file");
  struct arg_file *out = arg_file1(NULL, "out", "<filename_out>", "output file");
  struct arg_dbl *over = arg_dbl0(NULL, "over", "<F_over>", "oversample factor");
  struct arg_dbl *high = arg_dbl0(NULL, "high", "<F_high>", "highest-frequency factor");
  struct arg_int *dev = arg_int0(NULL, "device", "<device>", "device number");
  struct arg_end *end = arg_end(20);
  void *argtable[] = {in,out,over,high,dev,end};

  // Parse the command line

  int n_error = arg_parse(argc, argv, argtable);

  if (n_error == 0) {

    *filename_in = (char *) malloc(strlen(in->filename[0])+1);
    strcpy(*filename_in, in->filename[0]);

    *filename_out = (char *) malloc(strlen(out->filename[0])+1);
    strcpy(*filename_out, out->filename[0]);

    *F_over = over->count == 1 ? (float) over->dval[0] : 1.f;
    *F_high = high->count == 1 ? (float) high->dval[0] : 1.f;

    *device = dev->count == 1 ? dev->ival[0] : 0;

  }
  else {

    printf("Syntax: %s", argv[0]);
    arg_print_syntax(stdout, argtable, "\n");

    exit(EXIT_FAILURE);

  }

  // Finish

}


////
// CUDA Initialization
////

void
initialize_cuda (int device)
{ 
  
  // Select the device

  CUDA_CALL(cudaSetDevice(device));

  // Dummy call to initialize the CUDA runtime
  
  CUDA_CALL(cudaThreadSynchronize());

  // Finish

}


////
// Periodogram evaluation
////

void
eval_LS_periodogram (int N_t, int N_f, float df, float minf, 
		     float *t, float *X, float *P)
{


  // Allocate device memory and copy data over

  float *d_t;
  float *d_X;
  float *d_P;

  CUDA_CALL(cudaMalloc((void**) &d_t, N_t*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_X, N_t*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_P, N_f*sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_t, t, N_t*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_X, X, N_t*sizeof(float), cudaMemcpyHostToDevice));

  // Set up run parameters

  dim3 grid_dim(N_f/BLOCK_SIZE, 1, 1);
  dim3 block_dim(BLOCK_SIZE, 1, 1);

  //printf("Grid of %d frequency blocks of size %d threads\n", N_f/BLOCK_SIZE, BLOCK_SIZE);

  // Launch the kernel

  //printf("Launching kernel...\n");

  culsp_kernel<<<grid_dim, block_dim>>>(d_t, d_X, d_P, df, N_t, N_f, minf);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    fprintf(stderr, "Cuda error: kernel launch failed in file '%s' in line %i : %s.\n",
	    __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
  }

  CUDA_CALL(cudaThreadSynchronize());

  //printf("Completed!\n");

  // Copy data from the device

  CUDA_CALL(cudaMemcpy(P, d_P, N_f*sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(d_P));
  CUDA_CALL(cudaFree(d_X));
  CUDA_CALL(cudaFree(d_t));

  // Finish

}

void
bootstrap_LS_periodogram(int N_t, int N_f, float df, float minf, 
         float *t, float *X, float *max_heights, int N_bootstrap, int use_gpu_to_get_max){


  // Allocate device memory and copy data over

  float *d_t, *d_X, *d_P;
  float *P, *gmax;
  int i, gd, gdm, gdm0;
  float val;

  curandState *state;
  cudaError_t err; 

  CUDA_CALL(cudaMalloc((void**) &d_t, N_t*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_X, N_t*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_P, N_f*sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_t, t, N_t*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_X, X, N_t*sizeof(float), cudaMemcpyHostToDevice));

  // Get N_bootstraps LSP; then get max_height of each of these.
  // This can be made faster by altering the bootstrap code to get
  // the max within the function, though this is much more complicated
  // should be small speed increase: 
  //    <LC> ~ 0.4 MB; transfer time wasted = 2 * (0.4/1000 GB) / (~15 GB/s PCIe3x16) 
  //                 ~ 8E-4 seconds...maaaaaybe significant...
  // timing the results on 

  
  gd = N_f/BLOCK_SIZE;
  if (gd * BLOCK_SIZE < N_f) gd += 1; // ensure we have enough blocks

  dim3 grid_dim(gd, 1, 1);
  dim3 block_dim(BLOCK_SIZE, 1, 1);
  
  // setup the random generator
  CUDA_CALL(cudaMalloc((void **) &state, gd*BLOCK_SIZE * sizeof(curandState)));
  setup_curand_kernel<<<grid_dim, block_dim>>>(state, time(NULL));
  
  if (use_gpu_to_get_max){
    // allocate memory for the maximums array
    CUDA_CALL(cudaMalloc((void **) &gmax, gd * sizeof(float)));  

  } else {

    //printf("USING CPU TO FIND MAX(P_LS)\n");
    P = (float *) malloc(N_f * sizeof(float));
  }

  for(i=0; i<N_bootstrap; i++){

    bootstrap_kernel<<<grid_dim, block_dim>>>(d_t, d_X, d_P, df, N_t, N_f, minf, state);
    //CUDA_ERR_CHECK();

    if (use_gpu_to_get_max){
      // calculate the maximum.
      max_reduce<<<grid_dim, block_dim, BLOCK_SIZE * sizeof(float)>>>(d_P, gmax, N_f);
      
      // Now reduce until only one block is needed.
      gdm = gd;
      while (gdm > 1){

        gdm0 = gdm;
        gdm /= BLOCK_SIZE;
        if( gdm * BLOCK_SIZE < gdm0 ) gdm += 1;
        
        dim3 grid_dim_max(gdm, 1, 1);

        max_reduce<<<grid_dim_max, block_dim, BLOCK_SIZE*sizeof(float)>>>(gmax, gmax, gdm0);

      }

      //copy max(P) to the host
      CUDA_CALL(cudaMemcpy(&val, gmax, sizeof(float), cudaMemcpyDeviceToHost));
    
    } else {
    
      CUDA_CALL(cudaMemcpy(P, d_P, N_f*sizeof(float), cudaMemcpyDeviceToHost));
      //printf("CPUMAX");
      val = cpu_maxf(P, N_f); 
    }

    max_heights[i] = val;
  }

  //CUDA_ERR_CHECK();

  CUDA_CALL(cudaThreadSynchronize());

  CUDA_CALL(cudaFree(d_P));
  CUDA_CALL(cudaFree(d_X));
  CUDA_CALL(cudaFree(d_t));
  if (use_gpu_to_get_max) {
    CUDA_CALL(cudaFree(state));
    CUDA_CALL(cudaFree(gmax));
  }

  // Finish

}