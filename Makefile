# Makefile for DynamicGraphSimplified
# Compiles: main.cu + dsu.cu + sssp.cu + cpu_reference.cpp -> dynamic_graph
#
# Usage:
#   make           - build with auto-detected or default GPU arch
#   make SM=80     - build for sm_80 (A100); adjust for your cluster
#   make clean     - remove build artifacts
#
# On the cluster (compile on master/login node):
#   module purge
#   module load cuda/11.3          # or whichever CUDA module your cluster uses
#   make SM=80                     # sm_80 = NVIDIA A30 / A100
#   ./dynamic_graph dsu  data/dsu_test.txt  --init-frac 0.5 --batch-size 2 --num-batches -1
#   ./dynamic_graph sssp data/sssp_test.txt --source 0 --init-frac 0.5 --batch-size 2 --num-batches -1

NVCC     ?= nvcc
BUILD    := build

# Try to auto-detect SM from nvidia-smi; fall back to multi-arch PTX if unavailable.
SM ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.[:space:]' | grep -E '^[0-9]+$$')

ifeq ($(strip $(SM)),)
  GENCODE := -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_80,code=compute_80
else
  GENCODE := -gencode arch=compute_$(SM),code=sm_$(SM)
endif

# Optional: set NVCC_CCBIN to point nvcc at a compatible host compiler,
# or NVCC_CCBIN=UNSUPPORTED to pass -allow-unsupported-compiler.
NVCC_CCBIN ?=
ifeq ($(strip $(NVCC_CCBIN)),UNSUPPORTED)
  CCBIN_FLAG := -allow-unsupported-compiler
else ifneq ($(strip $(NVCC_CCBIN)),)
  CCBIN_FLAG := -ccbin $(NVCC_CCBIN)
else
  CCBIN_FLAG :=
endif

NVCCFLAGS := -O3 -std=c++17 -lineinfo $(GENCODE) $(CCBIN_FLAG) -Xcompiler -Wall

CU_SRCS  := main.cu dsu.cu sssp.cu
CPP_SRCS := cpu_reference.cpp
HEADER   := common.cuh

OBJS := $(patsubst %.cu,$(BUILD)/%.o,$(CU_SRCS)) \
        $(patsubst %.cpp,$(BUILD)/%.o,$(CPP_SRCS))

.PHONY: all clean

all: $(BUILD) dynamic_graph

dynamic_graph: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)
	@echo "Build successful: ./dynamic_graph"

$(BUILD)/%.o: %.cu $(HEADER) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/%.o: %.cpp $(HEADER) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) dynamic_graph results/*.txt results/*.csv
