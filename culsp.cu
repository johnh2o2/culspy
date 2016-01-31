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
#include <stdio.h>
#include <string.h>

#include "periodogram.h"
#include "culsp.h"
#include "culsp_kernel.cu"
#include "culsp_kernel_batch.cu"
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

// Main program
float *get_pinned_float_array(int n){
        float *x;
        cudaMallocHost((void **) &x, n * sizeof(float));
        return x;
}
float get_pinned_val(float *x,int i){
        return  x[i];
}
void set_pinned_val(float *x, int i, float val){
        x[i] = val;
}

void read_file_list(char *fname, char ***filenames, int *Nlc){
  FILE *file;
  *Nlc = get_nlines(fname);
  int i;

  *filenames = (char **)malloc(*Nlc * sizeof(char *));

  if((file = fopen(fname, "r")) == NULL) {
    printf("Unable to open file to read\n");
    exit(1);
  }

  for(i=0; i<*Nlc; i++){
    (*filenames)[i] = (char *)malloc(STRLEN * sizeof(char));
    fscanf(file, "%s ", (*filenames)[i]));
  }

  fclose(file);

}


int
main( int argc, char** argv) 
{

  Settings settings;
  char **lc_filenames;
  int *N_t;
  int Nlc, i;
  
  float *t;
  float *X;
  float *P;
  float freq, p;

  printf("initializing..\n");
  // Initialize
  initialize(argc, argv, &settings);

  printf("reading list..\n");

  // Read the list of light curves if there is one...
  if (settings.using_list){
    read_file_list(settings.filenames[INLIST], &lc_filenames, &Nlc);
  } else {
    // otherwise initialize the variables by hand
    Nlc = 1;
    lc_filenames = (char **)malloc(Nlc * sizeof(char *));
    lc_filenames[0] = (char *)malloc(STRLEN * sizeof(char));
    strcpy(lc_filenames[0], settings.filenames[IN]);
  }
  printf("done (%d lcs)\n", Nlc);
  printf("counting how much data is in each lc..\n");
  // Count total number of observations
  N_t = (int *)malloc(Nlc * sizeof(int));
  int offset = 0, total_size=0;
  for(i=0; i<Nlc; i++){
    N_t[i] = get_nlines(lc_filenames[i]);
    total_size += N_t[i];
  }
  printf("done. %d datapoints\n", total_size);

  float best_matches[2*Nlc];
  for (i=0; i<2*Nlc; i++ ) best_matches[i] = -1;

  CUDA_CALL(cudaMallocHost((void **) &t, total_size * sizeof(float)));
  CUDA_CALL(cudaMallocHost((void **) &X, total_size * sizeof(float)));
  CUDA_CALL(cudaMallocHost((void **) &P, Nlc * settings.Nfreq * sizeof(float)));


  // now read in the lightcurve data
  for(i=0; i<Nlc; i++){
    printf("reading lc %s..\n", lc_filenames[i]);
    read_light_curve(lc_filenames[i], N_t[i], &t[offset], &X[offset]);
    printf("    .. done (%d points)\n", N_t[i]);
    offset += N_t[i];
  }

  printf("initializing cuda ..\n");
  // Initialize CUDA
  initialize_cuda(settings.device);
		 
  printf("now computing LSP ..\n");
  // Evaluate the Lomb-Scargle periodogram
  compute_LSP_async(N_t, Nlc, &settings, t, X, P, best_matches);

  printf("done!\n");
  //// Write the data to file
  for(i=0; i<Nlc; i++){
    if (settings.only_get_max){
      offset = 2*i;
      freq = best_matches[offset];
      p = best_matches[offset+1];
      if (p > 0){
        printf("%-50s %-10.3e %-10.3e\n", lc_filenames[i], freq, p);
      }
    } /*else {
      char outname = 
      write_periodogram(filename_out, N_f, df, P);
    }*/
  }

  // Free up space

  cudaFreeHost(P);
  cudaFreeHost(X);
  cudaFreeHost(t);

  // Finish
  return 0;

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
/*

void
compute_LSP_batch (int *N_t, int Nlc, Settings settings,
         float *t, float *X, float *P)
{


  // Allocate device memory and copy data over
  int N_f = settings->Nfreq;
  float minf =settings->minf;
  float maxf = settings->maxf;
  float df = settings->df;

  float *d_t;
  float *d_X;
  float *d_P;
  int *d_N_t;
  int i, gd;
  int total_size=0;
  for (i=0; i<Nlc; i++) total_size += N_t[i];

  CUDA_CALL(cudaMalloc((void**) &d_t, total_size*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_X, total_size*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_P, N_f*Nlc*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_N_t, Nlc*sizeof(int)));

  CUDA_CALL(cudaMemcpy(d_N_t, N_t, Nlc*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_t, t, total_size*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_X, X, total_size*sizeof(float), cudaMemcpyHostToDevice));

  // Set up run parameters
  gd = Nlc * N_f/BLOCK_SIZE;
  while (gd * BLOCK_SIZE < Nlc*N_f) gd += 1;
  dim3 grid_dim(gd, 1, 1);
  dim3 block_dim(BLOCK_SIZE, 1, 1);

  //printf("Grid of %d frequency blocks of size %d threads\n", N_f/BLOCK_SIZE, BLOCK_SIZE);

  // Launch the kernel

  //printf("Launching kernel...\n");

  culsp_kernel_batch<<<grid_dim, block_dim>>>(d_t, d_X, d_P, df, d_N_t, Nlc, N_f, minf);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    fprintf(stderr, "Cuda error: kernel launch failed in file '%s' in line %i : %s.\n",
      __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
  }

  CUDA_CALL(cudaThreadSynchronize());

  //printf("Completed!\n");

  // Copy data from the device

  CUDA_CALL(cudaMemcpy(P, d_P, N_f*Nlc*sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(d_P));
  CUDA_CALL(cudaFree(d_X));
  CUDA_CALL(cudaFree(d_t));
  CUDA_CALL(cudaFree(d_N_t));
  // Finish

}
*/


void 
compute_LSP_async (int *N_t, int Nlc, Settings *settings,
         float *t, float *X, float *P, float *matches)
{

  
  int N_f = settings->Nfreq;
  float minf =settings->minf;
  float df = settings->df;
  int Nbootstraps = settings->Nbootstraps;
  int only_get_max = settings->only_get_max;
  int use_gpu_to_get_max = settings->use_gpu_to_get_max;

  float *d_t;
  float *d_X;
  float *d_P, *dp;

  cudaStream_t streams[Nlc];
  int total_size=0;
  int i, offset, gd, imax;
  float cmax;
  for (i=0; i<Nlc; i++) total_size += N_t[i];

  printf(" compute_LSP_async > here. setting up.\n");
  
  // Allocate device memory
  CUDA_CALL(cudaMalloc((void**) &d_t, total_size*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_X, total_size*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_P, N_f*sizeof(float)));
  

  // setup block/grid dimensions
  gd = N_f/BLOCK_SIZE;
  while (gd * BLOCK_SIZE < N_f) gd += 1;
  dim3 grid_dim(gd, 1, 1);
  dim3 block_dim(BLOCK_SIZE, 1, 1);

  printf(" compute_LSP_async > done.\n");
  // asynchronously transfer data and perform LSP on all lightcurves
  offset = 0;
  for(i=0; i < Nlc; i++){
    printf(" compute_LSP_async > started lc %d.\n", i);
    cudaStreamCreate(&streams[i]);

    CUDA_CALL(cudaMemcpyAsync(&d_t[offset], &t[offset], N_t[i] * sizeof(float), 
              cudaMemcpyHostToDevice, streams[i]));
    CUDA_CALL(cudaMemcpyAsync(&d_X[offset], &X[offset], N_t[i] * sizeof(float), 
              cudaMemcpyHostToDevice, streams[i]));
    
    
    culsp_kernel_stream<<<grid_dim, block_dim, 
                                0, streams[i]>>>( &d_t[offset], 
                                                  &d_X[offset],
                                                  &d_P[i*N_f], df, N_t[i], N_f, minf);

    if (!only_get_max){
      CUDA_CALL(cudaMemcpyAsync(&P[i*N_f], &d_P[i*N_f], N_f * sizeof(float), 
              cudaMemcpyDeviceToHost, streams[i]));   
    }
   
    offset += N_t[i];
  }
  printf(" compute_LSP_async > now we wait..\n");
  // wait for everyone
  CUDA_CALL( cudaDeviceSynchronize() );

  printf(" compute_LSP_async > done!\n");

  // check for problems
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    fprintf(stderr, "Cuda error: kernel launch failed in file '%s' in line %i : %s.\n",
      __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
  }


  // handle bootstrapping 
  if (Nbootstraps > 0){
    float max_heights[N_f], mu, std, maxp;
    int besti;

    offset = 0;
    for(i=0; i<Nlc; i++){
      // get a bootstrapped distribution of max(LSP) heights
      bootstrap_LSP(N_t[i], settings, &d_t[offset], &d_X[offset], max_heights);

      // now calculate the mean + std of that distro.
      cpu_stats(max_heights, N_f, &mu, &std);

      // find the highest peak of the LSP
      if (use_gpu_to_get_max || only_get_max){
        gpu_maxf_ind(&d_P[i*N_f], N_f, &maxp, &besti);
      } else {
        cpu_maxf_ind(&P[i*N_f], N_f, &maxp, &besti);
      }

      // -> SNR of that peak
      maxp = ((maxp - mu)/std);

      // if significant, record it
      if ( maxp > settings->cutoff){
        matches[2*i] = minf + df * besti; // freq
        matches[2*i + 1] = maxp;
        if (settings->verbose)
          printf("  PEAK FOUND! (%d) freq = %.4e ; snr = %.4e\n", 
                                    i, matches[2*i], matches[2*i+1]);
        
      }
      offset += N_t[i];
    }
  }


  CUDA_CALL(cudaFree(d_P));
  CUDA_CALL(cudaFree(d_X));
  CUDA_CALL(cudaFree(d_t));

  for (int i = 0; i < Nlc; ++i)
    CUDA_CALL( cudaStreamDestroy(streams[i]) );
  // Finish

}

void
bootstrap_LSP(int N_t, Settings *settings,
         float *d_t, float *d_X, float *max_heights){


  // Allocate device memory and copy data over

  float *d_P;
  float *P;
  int i, gd, imax;
  float val;
  float df = settings->df;
  float minf = settings->minf;
  int N_f = settings->Nfreq;
  int N_bootstrap = settings->Nbootstraps;

  curandState *state;

  CUDA_CALL(cudaMalloc((void**) &d_P, N_f*sizeof(float)));

  // Get N_bootstraps LSP; then get max_height of each of these.
  // This can be made faster by altering the bootstrap code to get
  // the max within the function, though this is much more complicated
  // should be small speed increase: 
  //    <LC> ~ 0.4 MB; transfer time wasted = 2 * (0.4/1000 GB) / (~15 GB/s PCIe3x16) 
  //                 ~ 8E-4 seconds...maaaaaybe significant...
  //          So I tried this and tested it -- cpu max is actually faster...

  
  gd = N_f/BLOCK_SIZE;
  if (gd * BLOCK_SIZE < N_f) gd += 1; // ensure we have enough blocks

  dim3 grid_dim(gd, 1, 1);
  dim3 block_dim(BLOCK_SIZE, 1, 1);
  
  // setup the random generator
  CUDA_CALL(cudaMalloc((void **) &state, gd*BLOCK_SIZE * sizeof(curandState)));
  setup_curand_kernel<<<grid_dim, block_dim>>>(state, time(NULL));
  
  if (!(settings->use_gpu_to_get_max)){
    P = (float *) malloc(N_f * sizeof(float));
  }

  for(i=0; i<N_bootstrap; i++){

    bootstrap_kernel<<<grid_dim, block_dim>>>(d_t, d_X, d_P, df, N_t, N_f, minf, state);
    //CUDA_ERR_CHECK();


    if (settings->use_gpu_to_get_max){
      gpu_maxf_ind(d_P, N_f, &val, &imax);

    } else {

      CUDA_CALL(cudaMemcpy(P, d_P, N_f*sizeof(float), cudaMemcpyDeviceToHost));
      val = cpu_maxf(P, N_f); 
    }

    max_heights[i] = val;
  }

  CUDA_CALL(cudaThreadSynchronize());

  CUDA_CALL(cudaFree(d_P));
  CUDA_CALL(cudaFree(state));

  // Finish

}
