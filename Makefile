################################################################################
#
# Build script for project
#
################################################################################
ARCH		:= sm_52

# Add source files here
EXECUTABLE      := culsp
# CUDA source files (compiled with cudacc)
CUFILES		:= culsp.cu
# CUDA dependency files
CU_DEPS		:= culsp_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES         := periodogram.cpp

################################################################################
# Rules and targets
NVCC=nvcc
CXX=g++
CC=gcc
#include ../../common/common.mk

#include /usr/src/nvidia-352-352.68/nvidia-modules-common.mk

cuda_lib=/usr/local/cuda/lib64
cuda_inc=/usr/local/cuda/include

BLOCK_SIZE=256

NVCCFLAGS := --ptxas-options=-v -DBLOCK_SIZE=$(BLOCK_SIZE) -arch $(ARCH)
CXXFLAGS := -DBLOCK_SIZE=$(BLOCK_SIZE)
LINK := -largtable2 -lm -lcudart -L$(cuda_lib)

all : $(EXECUTABLE)


$(EXECUTABLE): culsp.o periodogram.o
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $^ $(LINK) 

periodogram.o : periodogram.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

culsp.o : culsp.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $^ -I$(cuda_inc)
