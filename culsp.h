#ifndef CULSP_H
#define CULSP_H
void initialize(int, char **, char **, char **, float *, float *, int *);
void initialize_cuda(int);
void eval_LS_periodogram(int, int, float, float, float *, float *, float *);
#endif
