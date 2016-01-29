

void
stream_eval_LS_periodogram (int *N_t, int Nlc, int N_f, float df, float minf, 
         float *t, float *X, float *P)
{

  // Allocate device memory and copy data over

  float *d_t;
  float *d_X;
  float *d_P;

  cudaStream_t streams[Nlc];

  int total_size=0, i,offset, gd;
  for (i=0; i<Nlc; i++) total_size += N_t[i];

  CUDA_CALL(cudaMalloc((void**) &d_t, total_size*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_X, total_size*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**) &d_P, N_f*Nlc*sizeof(float)));

  gd = N_f/BLOCK_SIZE;
  while (gd * BLOCK_SIZE < N_f) gd += 1;
  dim3 grid_dim(gd, 1, 1);
  dim3 block_dim(BLOCK_SIZE, 1, 1);

  offset = 0;
  for(i=0; i < Nlc; i++){
    cudaStreamCreate(&streams[i])

    CUDA_CALL(cudaMemcpyAsync(d_t[offset], t[offset], N_t[i] * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyAsync(d_X[offset], X[offset], N_t[i] * sizeof(float), cudaMemcpyHostToDevice));
    
    culsp_kernel_stream<<<grid_dim, block_dim, 0, streams[i]>>>(d_t, d_X, d_P, df, offset, 
                                                                            N_t[i], N_f, minf);

    CUDA_CALL(cudaMemcpyAsync(P[i*N_f], d_P[i*N_f], N_f * sizeof(float), cudaMemcpyDeviceToHost));
    offset += N_t[i];
  }

  CUDA_CALL( cudaDeviceSynchronize() );

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    fprintf(stderr, "Cuda error: kernel launch failed in file '%s' in line %i : %s.\n",
      __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
  }

  CUDA_CALL(cudaFree(d_P));
  CUDA_CALL(cudaFree(d_X));
  CUDA_CALL(cudaFree(d_t));

  for (int i = 0; i < Nlc; ++i)
    CUDA_CALL( cudaStreamDestroy(streams[i]) );
  // Finish

}