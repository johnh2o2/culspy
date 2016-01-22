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

python_inc=/usr/include/python2.7 
BLOCK_SIZE=256

NVCCFLAGS := -Xcompiler -fpic --ptxas-options=-v -DBLOCK_SIZE=$(BLOCK_SIZE) -arch $(ARCH)
CXXFLAGS := -fPIC -DBLOCK_SIZE=$(BLOCK_SIZE)
LINK := -largtable2 -lm -lcudart -L$(cuda_lib)

all : $(EXECUTABLE)


$(EXECUTABLE): culsp.o periodogram.o
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $^ $(LINK) 

periodogram.o : periodogram.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

periodogram_nomain.o : periodogram.cpp
	$(CXX) -Dmain=oldmain $(CXXFLAGS) -c -o $@ $^

culsp.o : culsp.cu
	$(NVCC) -Xcompiler -fpic $(NVCCFLAGS) -c -o $@ $^ -I$(cuda_inc)

culsp_wrap.o : culsp_wrap.cpp
	$(CXX) -fPIC $(CXXFLAGS) -c -o $@ $^ -I$(python_inc)

python : culsp_wrap.o culsp.o periodogram_nomain.o
	$(CXX) -fPIC $(CXXFLAGS) -shared -o _culspy.so $^ $(LINK) -lpython2.7 
	#mkdir culspy
	#mv culspy.py culspy/
	#mv _culspy.so culspy/
	#touch culspy/__init__.py

clean :
	rm -f *o *so *pyc $(EXECUTABLE)
	rm -f -r culspy/
	rm -r -f build/
