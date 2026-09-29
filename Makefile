# PALMA ORLANDO 0622702433 p.orlando8@studenti.unisa.it
# Course: High Performance Computing 2025/2026
# Lecturer: Francesco Moscato fmoscato@unisa.it
#
# Copyright (C) 2025 Palma Orlando
#
# This file is part of ManberMyers_Orlando_Palma_HPC_IZ.
#
# ManberMyers_Orlando_Palma_HPC_IZ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ManberMyers_Orlando_Palma_HPC_IZ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ManberMyers_Orlando_Palma_HPC_IZ.  If not, see <http://www.gnu.org/licenses/>.

# ==============================================================================
# MAKEFILE - Manber-Myers Parallel Suffix Array
# Purpose: Automated build, test, and benchmark suite for sequential, MPI, and
#          CUDA implementations of the Manber-Myers suffix array algorithm.
#
# Requirements:
#   - gcc (with OpenMP support: -fopenmp)
#   - mpicc (via WSL for Windows, native on Linux)
#   - nvcc (NVIDIA CUDA Compiler)
#   - Python 3.x (for test automation and plotting)
#
# Main Targets:
#   all      - Setup environment and compile data generator
#   test     - Run complete benchmark suite (compile + execute + plot)
#   clean    - Remove all generated files (binaries, data, results)
#   data     - Generate input test files
#   help     - Display usage information
# ==============================================================================

# Directory Structure
SRC_DIR := Source
HDR_DIR := Headers
BUILD_DIR := Build
DATA_DIR := Data
RESULTS_DIR := Results
DOC_DIR := Documentation

# Python Scripts
BUILD_MANAGER := $(SRC_DIR)/Scripts/build_manager.py
PLOTS_SCRIPT := $(SRC_DIR)/Scripts/plots.py
GEN_SOURCE := $(SRC_DIR)/Scripts/string_generator.c
GEN_EXE := $(BUILD_DIR)/string_generator

# Detect OS for portable commands
ifeq ($(OS),Windows_NT)
    RM := cmd /c rmdir /s /q
    MKDIR := cmd /c if not exist
    MKDIR_END := mkdir
    EXE_EXT := .exe
    GEN_EXE := $(BUILD_DIR)/string_generator.exe
else
    RM := rm -rf
    MKDIR := mkdir -p
    MKDIR_END :=
    EXE_EXT :=
endif

# PHONY TARGETS (non-file targets)
.PHONY: all clean test data dirs generator help plots-enhanced

# MAIN TARGETS 
# 1. ALL: Setup environment
all: dirs generator
	@echo "----------------------------------------------------------------"
	@echo " Environment Ready!"
	@echo "----------------------------------------------------------------"
	@echo " Next steps:"
	@echo "   1. Run 'make data' to generate test files"
	@echo "   2. Run 'make test' to execute full benchmark suite"
	@echo "----------------------------------------------------------------"

# 2. TEST: Complete benchmark suite
# - Compiles all versions (Sequenziale, MPI, CUDA)
# - Executes tests on all data files
# - Generates performance plots
test: dirs
	@echo "----------------------------------------------------------------"
	@echo " Starting Complete Test Suite"
	@echo "----------------------------------------------------------------"
	@echo ""
	@echo "[1/6] Testing: string_1MB.bin"
	python $(BUILD_MANAGER) string_1MB.bin
	@echo ""
	@echo "[2/6] Testing: string_50MB.bin"
	python $(BUILD_MANAGER) string_50MB.bin
	@echo ""
	@echo "[3/6] Testing: string_100MB.bin"
	python $(BUILD_MANAGER) string_100MB.bin
	@echo ""
	@echo "[4/6] Testing: string_200MB.bin"
	python $(BUILD_MANAGER) string_200MB.bin
	@echo ""
	@echo "[5/6] Testing: string_500MB.bin"
	python $(BUILD_MANAGER) string_500MB.bin
	@echo ""
	@echo "[6/6] Generating Performance Plots..."
	python $(PLOTS_SCRIPT)
	@echo ""
	@echo "----------------------------------------------------------------"
	@echo " Test Suite Completed Successfully!"
	@echo "----------------------------------------------------------------"
	@echo " Results available in: $(RESULTS_DIR)/"
	@echo " Plots available in: $(RESULTS_DIR)/plots/"
	@echo "----------------------------------------------------------------"
	
# 3. CLEAN: Remove all generated files
clean:
	@echo "----------------------------------------------------------------"
	@echo " Cleaning Project..."
	@echo "----------------------------------------------------------------"
ifeq ($(OS),Windows_NT)
	@if exist $(BUILD_DIR) $(RM) $(BUILD_DIR)
	@if exist $(RESULTS_DIR) $(RM) $(RESULTS_DIR)
	@if exist $(DATA_DIR) $(RM) $(DATA_DIR)
else
	@$(RM) $(BUILD_DIR) $(RESULTS_DIR) $(DATA_DIR)
endif
	@echo " Cleanup completed."
	@echo "----------------------------------------------------------------"


# UTILITY TARGETS
# Create directory structure
dirs:
ifeq ($(OS),Windows_NT)
	@$(MKDIR) $(BUILD_DIR) $(MKDIR_END) $(BUILD_DIR)
	@$(MKDIR) $(DATA_DIR) $(MKDIR_END) $(DATA_DIR)
	@$(MKDIR) $(RESULTS_DIR) $(MKDIR_END) $(RESULTS_DIR)
	@$(MKDIR) $(DOC_DIR) $(MKDIR_END) $(DOC_DIR)
else
	@$(MKDIR) $(BUILD_DIR) $(DATA_DIR) $(RESULTS_DIR) $(DOC_DIR)
endif

# Compile data generator
generator: dirs
	@echo "----------------------------------------------------------------"
	@echo " Compiling Data Generator..."
	@echo "----------------------------------------------------------------"
	gcc $(GEN_SOURCE) -o $(GEN_EXE)
	@echo " Generator compiled: $(GEN_EXE)"

# Generate test data files
data: generator
	@echo "----------------------------------------------------------------"
	@echo " Generating Input Data Files..."
	@echo "----------------------------------------------------------------"
	$(GEN_EXE)
	@echo ""
	@echo " Data files generated in: $(DATA_DIR)/"
	@echo "----------------------------------------------------------------"

# Display help information
help:
	@echo "----------------------------------------------------------------"
	@echo " Manber-Myers HPC - Makefile Help"
	@echo "----------------------------------------------------------------"
	@echo ""
	@echo " Available targets:"
	@echo ""
	@echo "   make all      - Setup environment (create dirs + compile generator)"
	@echo "   make data     - Generate test input files (1, 50, 100, 200, 500 MB)"
	@echo "   make test     - Run full benchmark suite (compile + test + plot)"
	@echo "   make clean    - Remove all generated files"
	@echo "   make help     - Display this help message"
	@echo ""
	@echo " Typical workflow:"
	@echo "   1. make all       # First time setup"
	@echo "   2. make data      # Generate test files"
	@echo "   3. make test      # Run benchmarks"
	@echo ""
	@echo " Requirements:"
	@echo "   - GCC with OpenMP support"
	@echo "   - MPI compiler (mpicc via WSL on Windows)"
	@echo "   - NVIDIA CUDA Toolkit (nvcc)"
	@echo "   - Python 3.x with matplotlib, seaborn, pandas"
	@echo ""
	@echo "----------------------------------------------------------------"