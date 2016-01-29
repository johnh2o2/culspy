#ifndef CULSP_H
#define CULSP_H
void initialize(int, char **, char **, char **, float *, float *, int *);
void initialize_cuda(int);
void eval_LS_periodogram(int, int, float, float, float *, float *, float *);
void batch_eval_LS_periodogram(int*, int, int, float, float, float *, float *, float *);
void stream_eval_LS_periodogram(int*, int, int, float, float, float *, float *, float *);
void bootstrap_LS_periodogram(int, int, float, float, float *, float *, float *, int, int);
#endif
