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

#ifndef _PERIODOGRAM_H_
#define _PERIODOGRAM_H_

void read_light_curve (char *, int *, float **, float **);
int get_nlines(char *filename);
void initialize (int argc, char **argv, Settings *settings);
void write_periodogram (char *, int, float, float *);
void set_frequency_params (int, float *, float, float, int *, float *);
double get_time ();

#endif
