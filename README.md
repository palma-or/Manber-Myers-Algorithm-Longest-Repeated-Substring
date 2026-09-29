# Manber-Myers Parallel Suffix Array - HPC Implementation

**Student:** Orlando Palma 
**Course:** High Performance Computing 2025/2026  
**Lecturer:** Prof. Francesco Moscato - fmoscato@unisa.it

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Requirements](#-requirements)
3. [Directory Structure](#-directory-structure)
4. [Compilation](#-compilation)
5. [Usage](#-usage)
6. [Running Tests](#-running-tests)
7. [Results Interpretation](#-results-interpretation)
8. [License](#-license)

---

## Project Overview

This project implements the **Manber-Myers Suffix Array** construction algorithm with three approaches:

1.  **Sequential Version** - Baseline implementation (O(n log n)).
2.  **MPI Version** - Distributed memory parallelization using Message Passing Interface.
3.  **CUDA Version** - GPU acceleration for massive parallel sorting.

The project includes:
* Automated build and test suite.
* Performance benchmarking across multiple input sizes (1MB to 500MB).
* Speedup analysis and visualization tools.
* Longest Repeated Substring (LRS) computation.

---

## Requirements

### Software Dependencies

#### Compilers
* **GCC** (version 15.2.0)
* **MPI Compiler** (mpicc versione 13.3.0)
    * *Windows:* Install via WSL (Windows Subsystem for Linux).
* **NVIDIA CUDA Toolkit** (version 11.8.89)

#### Python Environment
* Python 3.8+ with packages:
 ```bash
    pip install matplotlib seaborn pandas numpy
```

### Hardware Requirements
* **CPU:** Multi-core processor (tested on 2-4 cores).
* **GPU:** NVIDIA GPU with at least 2GB VRAM.
* **RAM:** Minimum 8GB.
* **Disk:** ~2GB free space for test data and results.

---

## Directory Structure

```
ManberMyers_Orlando_Palma_HPC_IZ/
│
├── Source/                     # Source code files
│   ├── seq_main.c              # Sequential main
│   ├── seq_suffix_array.c      # Sequential Manber-Myers
│   ├── mpi_main.c              # MPI main
│   ├── mpi_suffix_array.c      # MPI parallel logic
│   ├── cuda_main.cu            # CUDA main
│   ├── cuda_suffix_array.cu    # CUDA kernels
│   ├── common_functions.c      # Shared utilities
│   ├── merge_algorithms.c      # MPI merge logic
│   └── Scripts/ 
│       ├── build_manager.py    # Automated compilation/testing
│       ├── plots.py            # Performance visualization
│       └── string_generator.c  # Test data generator
│
├── Headers/                    # Header files
│   ├── seq_suffix_array.h
│   ├── mpi_suffix_array.h
│   ├── cuda_suffix_array.h
│   ├── common_functions.h
│   └── merge_algorithms.h
│
├── Build/                      # Compiled binaries (auto-generated)
│   ├── Sequenziale/
│   ├── MPI/
│   └── CUDA/
│
├── Data/                       # Test input files (auto-generated)
│   ├── string_1MB.bin
│   ├── string_50MB.bin
│   ├── string_100MB.bin
│   ├── string_200MB.bin
│   └── string_500MB.bin
│
├── Results/                    # Benchmark results (auto-generated)
│   ├── results_string_1MB.bin.json
│   ├── ...
│   └── plots/                  # Performance graphs
│
├── Documentation/              # Project documentation
│   └── report.pdf              # Performance analysis report
│
├── Makefile                    # Build automation
├── LICENSE                     # GPL-3.0 License
└── README.md                   # This file
```

---

## Compilation

### Quick Start (Automated)

The Makefile handles all compilation automatically:

```bash
# 1. Setup environment
make all

# 2. Generate test data
make data

# 3. Compile and run benchmarks
make test
```

### Manual Compilation
### Compilation Note for Windows Users
To compile the CUDA version on Windows, you **must** run the compilation commands (or `make`) inside the **x64 Native Tools Command Prompt for VS 2022** (or 2019). This ensures `nvcc` can locate the MSVC host compiler (`cl.exe`).

#### Sequential Version
```bash
gcc -I Headers Source/seq_main.c Source/seq_suffix_array.c \
    Source/common_functions.c -fopenmp -O3 -o Build/Sequenziale/seq
```

#### MPI + OpenMP Version
```bash
# On Linux/WSL
mpicc -I Headers Source/mpi_main.c Source/mpi_suffix_array.c \
      Source/merge_algorithms.c Source/seq_suffix_array.c \
      -fopenmp -O3 -o Build/MPI/mpi_prog
```

#### CUDA + OpenMP Version
```bash
nvcc -I Headers Source/cuda_main.cu Source/cuda_suffix_array.cu \
     Source/seq_suffix_array.c Source/common_functions.c \
     -Xcompiler -fopenmp -O3 -DBLOCK_SIZE=256 -o Build/CUDA/cuda_prog
```

---

## Usage

### Running Individual Programs

#### Sequential
```bash
./Build/Sequenziale/seq Data/string_1MB.bin
```

#### MPI (4 processes)
```bash
# On Linux/WSL
mpirun -np 4 ./Build/MPI/mpi_prog Data/string_1MB.bin
```

#### CUDA
```bash
./Build/CUDA/cuda_prog Data/string_1MB.bin
```

### Command-Line Options

All programs accept a single argument: the input binary file path.

**Example Output:**
```
SUFFIX ARRAY - VERSIONE SEQUENZIALE 
------- RISULTATI ------- 
INFORMAZIONI FILE:
   File:                     random_strings/string_1MB.bin
   Dimensione:               49932 bytes (0.05 MB)
   LRS:                      4 caratteri

TEMPI DI ESECUZIONE:
   Suffix Array:             0.0290 s  [96.7%]
   LCP Array:                0.0010 s  [3.3%]
   LRS Search:               0.0000 s  [0.0%]
   --------------------------------------------------
   Totale:                   0.0300 s

MEMORIA:
   Testo:                    199728 bytes (0.19 MB)
   Suffix Array:             199728 bytes (0.19 MB)
   LCP Array:                199728 bytes (0.19 MB)
   --------------------------------------------------
   Totale:                   599184 bytes (0.57 MB)

```

---

## Running Tests

### Full Benchmark Suite

Execute complete performance evaluation:

```bash
make test
```

This will:
1. Compile all versions (Sequential, MPI, CUDA) with optimizations (-O0 to -O3)
2. Test on all data sizes (1MB, 50MB, 100MB, 200MB, 500MB)
3. Execute 3 runs per configuration (averaging results)
4. Generate performance plots in `Results/plots/`

### Custom Test (Single File)

```bash
python Source/Scripts/build_manager.py string_1MB.bin
```

### Cleaning

Remove all generated files:
```bash
make clean
```

---

## 📊 Results Interpretation

### Output Files

1. **JSON Results** (`Results/results_string_XMB.bin.json`)
   - Raw timing data for all configurations
   - Includes: total time, SA construction, LCP, LRS phases

2. **Performance Plots** (`Results/plots/`)
   - `Time_Bar_O{0-3}.png` - Execution time comparisons
   - `Speedup_Bar_O{0-3}.png` - Speedup vs Sequential baseline

### Understanding Speedup

**Speedup = T_sequential / T_parallel**

- Speedup > 1: Faster than sequential
- Speedup = 1: Same speed as sequential  
- Speedup < 1: Slower than sequential (overhead dominated)

---

## Algorithm Details

### Manber-Myers Overview

The Manber-Myers algorithm constructs suffix arrays in **O(n log n)** time by:

1. **Initialization (h=1)**: Sort suffixes by first 2 characters
2. **Doubling (h=2^k)**: Iteratively double comparison length
3. **Rank Update**: Assign new ranks based on current ordering
4. **Termination**: Stop when all suffixes have unique ranks

### Parallelization Strategies

#### MPI Approach
- **Domain Decomposition**: Split text across processes
- **Local Sorting**: Each process computes local suffix array
- **Iterative Merge**: Logarithmic merge of sorted arrays

#### CUDA Approach
- **CPU Suffix Array**: Manber-Myers algorithm on CPU 
- **CPU LCP**: Kasai algorithm on CPU (better memory access)
- **GPU LRS**: Parallel reduction to find maximum LCP value

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**.

```
Copyright (C) 2025 Palma Orlando

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
```

Full license text: [LICENSE](LICENSE) file or https://www.gnu.org/licenses/gpl-3.0.html

---
