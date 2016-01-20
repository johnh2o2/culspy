################################################################################
#
# Build script for project
#
################################################################################

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

include ../../common/common.mk

BLOCK_SIZE=256

NVCCFLAGS += --ptxas-options=-v -DBLOCK_SIZE=$(BLOCK_SIZE)
CXXFLAGS += -DBLOCK_SIZE=$(BLOCK_SIZE)
LINK += -largtable2
