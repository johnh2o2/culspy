#ifndef CULSP_H
#define CULSP_H

#define STRLEN 500

enum Filename { INLIST, OUT, IN};

typedef struct {
  char filenames[3][STRLEN];
  int Nfreq, Nbootstraps, device, using_list;
  float minf, maxf, cutoff, F_over, F_high, df; 
  int use_gpu_to_get_max, verbose, only_get_max;
  
} Settings;

void initialize_cuda(int);
void eval_LS_periodogram(int, int, float, float, float *, float *, float *);
//void batch_eval_LS_periodogram(int*, int, int, float, float, float *, float *, float *);
void compute_LSP_async(int*, int, Settings *, float*, float*, float *, float *);
void bootstrap_LSP(int, Settings *, float *, float *, float *);
#endif
