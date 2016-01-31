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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <argtable2.h>

#include "periodogram.h"

////
// Read a light curve from a file
////

int get_nlines(char *filename){
  int nlines = 0, c;
  FILE *file;

  if((file = fopen(filename, "r")) == NULL) {
    printf("Unable to open file to read\n");
    exit(1);
  }
  
  while ((c = fgetc(file)) != EOF) {
    if (c == '\n') nlines++;
  }

  fclose(file);

  return nlines;
}

void
read_light_curve (char *filename, int N_t, float *t, float *X)
{

  // Open the file 


  FILE *file;

  printf("Reading light curve from file %s\n", filename);

  if((file = fopen(filename, "r")) == NULL) {
    printf("Unable to open file to read\n");
    exit(1);
  }

  double *td = (double *) malloc(N_t*sizeof(double));
  double *Xd = (double *) malloc(N_t*sizeof(double));

  // Read in the data

  for(int j = 0; j < N_t; j++) {
    fscanf(file, "%lf %lf", td+j, Xd+j);
  }

  fclose(file);

  printf("Read in %d points in %s.\n", N_t, filename);

  // Shift the time to have zero mean

  double t_mean = 0.;

  for(int j = 0; j < N_t; j++) {
    t_mean += td[j];
  }

  t_mean /= N_t;

  for(int j = 0; j < N_t; j++) {
    t[j] = td[j] - t_mean;
  }

  // Shift the data to have zero mean, and scale it to have unit
  // variance (required by L-S algorithm)

  double X_mean = 0.;
  double XX_mean = 0.;

  for(int j = 0; j < N_t; j++) {
    X_mean += Xd[j];
    XX_mean += Xd[j]*Xd[j];
  }

  X_mean /= N_t;
  XX_mean /= N_t;
  
  float rsqrt_var = 1./sqrt((XX_mean - X_mean*X_mean)*(N_t)/(N_t-1));

  for(int j = 0; j < N_t; j++) {
    X[j] = (Xd[j]-X_mean)*rsqrt_var;
  }

  // Finish

  free(Xd);
  free(td);

}


////
// Write a periodogram to a file
////

void
write_periodogram (char *filename, int N_f, float df, float *P)
{
 
  // Open the file 

  FILE *file;

  printf("Writing to file %s\n", filename);
 
  if((file = fopen(filename, "w")) == NULL) {
    printf("Unable to open file to write\n");
    exit(1);
  }

  // Write the data

  for(int j = 0; j < N_f; j++) {
    fprintf(file, "%16.8E %16.8E\n", (j+1)*df, P[j]);
  }

  fclose(file);

  // Finish

}


////
// Set up frequency parameters
////

void
set_frequency_params (int N_t, float *t, float F_over, float F_high, int *N_f, float *df) 
{

  // Calculate the observation time span

  float t_min = t[0];
  float t_max = t[0];

  for(int j = 1; j < N_t; j++) {
    t_min = t[j] < t_min ? t[j] : t_min;
    t_max = t[j] > t_max ? t[j] : t_max;
  }

  float del_t = t_max - t_min;

  //printf("Average time spacing: %f\n", del_t/N_t);

  // Set up the frequency spacing

  *df = 1.f/(F_over*del_t);

  //printf("Frequency spacing: %f\n", *df);

  // Set up the maximum frequency

  float f_max = F_high*N_t/(2*del_t);

  //printf("Maximum frequency: %f\n", f_max);

  // Set up the number of frequency points

  *N_f = (int) (f_max/(*df)) + 1;

  #ifdef BLOCK_SIZE
  *N_f += (BLOCK_SIZE - *N_f % BLOCK_SIZE) % BLOCK_SIZE;
  #endif

  //printf("Number of frequencies: %d\n", *N_f);

  // Finish

}


////
// Initialization
////



void
initialize (int argc, char **argv, Settings *settings)
{

  // Set up the argtable structs

  
  struct arg_int *nbootstraps = arg_int0(NULL, "nboots", "<N_bootstraps>", 
                                            "number of bootstrapped samples to compute");
  struct arg_int *nfreq = arg_int1(NULL, "nfreq", "<N_frequencies>", 
                                            "number of frequencies to search");

  struct arg_int *gpumax = arg_int0(NULL, "gpumax", "<use_gpu_to_get_max>", 
                                            "Use a GPU-reduce kernel to get the max of the LSP");
  struct arg_dbl *minf = arg_dbl1(NULL, "minf", "<min_frequency>", 
                                            "minimum freq to search");
  struct arg_dbl *maxf = arg_dbl1(NULL, "maxf", "<max_frequency>", 
                                            "maximum freq to search");
  struct arg_dbl *cutoff = arg_dbl1(NULL, "cutoff", "<cutoff_std>", 
                                            "minimum SNR for peak of LSP");

  struct arg_file *in = arg_file0(NULL, "in", "<filename_in>", "input file");
  struct arg_file *inlist = arg_file0(NULL, "list_in", "<list_in>", "list of input files");
  struct arg_file *out = arg_file1(NULL, "out", "<filename_out>", "output file");
  struct arg_dbl *over = arg_dbl0(NULL, "over", "<F_over>", "oversample factor");
  struct arg_dbl *high = arg_dbl0(NULL, "high", "<F_high>", "highest-frequency factor");
  struct arg_int *dev = arg_int0(NULL, "device", "<device>", "device number");

  struct arg_end *end = arg_end(20);
  void *argtable[] = {inlist, nbootstraps, nfreq, minf, maxf, cutoff,
                        in, out, over, high, dev, end};

  // Parse the command line

  int n_error = arg_parse(argc, argv, argtable);

  if (n_error == 0) {

    if (inlist->count == 1){
      strcpy(settings->filenames[INLIST], inlist->filename[0]);
      settings->using_list = 1;
    }
    else { settings->using_list = 0; }

    if (in->count == 1){
      strcpy(settings->filenames[IN], in->filename[0]);
      if (settings->using_list == 1){
        fprintf(stderr, "both <in> and <inlist> are specified!");
        exit(EXIT_FAILURE);
      }
    }

    strcpy(settings->filenames[OUT], out->filename[0]);

    settings->F_over = over->count == 1 ? (float) over->dval[0] : 1.f;
    settings->F_high = high->count == 1 ? (float) high->dval[0] : 1.f;

    settings->Nfreq = nfreq->ival[0];
    settings->Nbootstraps = nbootstraps->count == 1 ? nbootstraps->ival[0] : 0;
    settings->device = dev->ival[0];

    settings->minf = (float) minf->dval[0];
    settings->maxf = (float) maxf->dval[0];

    settings->df = (settings->maxf - settings->minf)/settings->Nfreq;
    settings->cutoff = (float) cutoff->dval[0];

    settings->use_gpu_to_get_max = gpumax->count == 1 ? gpumax->ival[0] : 0;
    settings->verbose = 1; // TODO: add this as commandline argument
    settings->only_get_max = 1; // TODO: add this as commandline argument

  }
  else {

    printf("Syntax: %s", argv[0]);
    arg_print_syntax(stdout, argtable, "\n");

    exit(EXIT_FAILURE);

  }

  // Finish

}
////
// Get elapsed time
////

double
get_time () 
{

  struct timeval tp;
  int i;

  i = gettimeofday(&tp, NULL);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );

}
