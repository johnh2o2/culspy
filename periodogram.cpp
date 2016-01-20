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

#include "periodogram.h"

////
// Read a light curve from a file
////

void
read_light_curve (char *filename, int *N_t, float **t, float **X)
{

  // Open the file 

  FILE *file;

  printf("Reading light curve from file %s\n", filename);

  if((file = fopen(filename, "r")) == NULL) {
    printf("Unable to open file to read\n");
    exit(1);
  }

  // Count the number of lines

  *N_t = 0;
  double dummy;

  while (fscanf(file, "%lf %lf", &dummy, &dummy) != EOF) {
    (*N_t)++;
  }

  // Allocate arrays (double-precision buffers to avoid roundoff)

  *t = (float *) malloc(*N_t*sizeof(float));
  *X = (float *) malloc(*N_t*sizeof(float));
  
  double *td = (double *) malloc(*N_t*sizeof(double));
  double *Xd = (double *) malloc(*N_t*sizeof(double));

  // Read in the data

  rewind(file);

  for(int j = 0; j < *N_t; j++) {
    fscanf(file, "%lf %lf", td+j, Xd+j);
  }

  fclose(file);

  printf("Read in %d points\n", *N_t);

  // Shift the time to have zero mean

  double t_mean = 0.;

  for(int j = 0; j < *N_t; j++) {
    t_mean += td[j];
  }

  t_mean /= *N_t;

  for(int j = 0; j < *N_t; j++) {
    (*t)[j] = td[j] - t_mean;
  }

  // Shift the data to have zero mean, and scale it to have unit
  // variance (required by L-S algorithm)

  double X_mean = 0.;
  double XX_mean = 0.;

  for(int j = 0; j < *N_t; j++) {
    X_mean += Xd[j];
    XX_mean += Xd[j]*Xd[j];
  }

  X_mean /= *N_t;
  XX_mean /= *N_t;
  
  float rsqrt_var = 1./sqrt((XX_mean - X_mean*X_mean)*(*N_t)/(*N_t-1));

  for(int j = 0; j < *N_t; j++) {
    (*X)[j] = (Xd[j]-X_mean)*rsqrt_var;
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

  printf("Average time spacing: %f\n", del_t/N_t);

  // Set up the frequency spacing

  *df = 1.f/(F_over*del_t);

  printf("Frequency spacing: %f\n", *df);

  // Set up the maximum frequency

  float f_max = F_high*N_t/(2*del_t);

  printf("Maximum frequency: %f\n", f_max);

  // Set up the number of frequency points

  *N_f = (int) (f_max/(*df)) + 1;

  #ifdef BLOCK_SIZE
  *N_f += (BLOCK_SIZE - *N_f % BLOCK_SIZE) % BLOCK_SIZE;
  #endif

  printf("Number of frequencies: %d\n", *N_f);

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
