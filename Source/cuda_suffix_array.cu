/* PALMA ORLANDO 0622702433 p.orlando8@studenti.unisa.it
 * Course: High Performance Computing 2025/2026
 * Lecturer: Francesco Moscato   fmoscato@unisa.it
 *
 * Copyright (C) 2025 Palma Orlando
 *
 * This file is part of ManberMyers_Orlando_Palma_HPC_IZ.
 *
 * ManberMyers_Orlando_Palma_HPC_IZ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ManberMyers_Orlando_Palma_HPC_IZ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ManberMyers_Orlando_Palma_HPC_IZ.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "cuda_suffix_array.h"
#include "seq_suffix_array.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // Valore di default 
#endif

#define CUDA_CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n", \
                __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


/* KERNEL GPU: reduce_max_kernel
 * Viene eseguito in parallelo da ogni thread sulla GPU e implementa l'algoritmo 
 * di riduzione parallela per trovare il massimo all'interno del proprio blocco.
 * __global__ --> kernel che viene lanciato dalla CPU ed eseguito sulla GPU.
 * extern __shared__ int sdata[] --> sdata è allocato in memoria condivisa per ogni blocco.
 * - tid: è l'ID locale del thread all'interno del blocco 
 * - i: è l'ID globale del thread e corrisponde all'indice dell'elemento nell'array 
 * LCP globale (d_input[i]).
 * 
 * Ogni thread carica un elemento (d_input[i]) dalla memoria globale alla 
 * Shared Memory. Il controllo i < n impedisce accessi fuori limite e garantisce
 * che tutti i thread del blocco abbiano completato il caricamento prima 
 * che inizi la fase di riduzione.
 * Successivamente, viene eseguita la riduzione binaria per cui, ad ogni 
 * iterazione, la distanza di confronto (s) viene dimezzata (s >>= 1).
 * I thread con tid < s confrontano il loro valore con il valore del 
 * thread sdata[tid + s] e si mantiene il valor massimo.
 * Infine, il thread master di ogni blocco (tid == 0) scrive il risultato
 * finale del blocco nell'array globale d_output[blockIdx.x]
 */
__global__ void reduce_max_kernel(int *d_input, int *d_output, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? d_input[i] : 0;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) d_output[blockIdx.x] = sdata[0];
}


// FUNZIONE: findLRSCUDA
int findLRSCUDA(int *lcp, int n) {
    int threads = BLOCK_SIZE;
    // Calcola il numero di blocchi necessari per coprire la dimensione dell'LCP Array
    int blocks = (n + threads - 1) / threads;

    // Si allocano sulla Device Memory l'array d_lcp che è la copia dell'LCP Array e 
    // d_partial_max che è l'array di output che conterrà i massimi parziali
    int *d_lcp, *d_partial_max;
    CUDA_CHECK(cudaMalloc(&d_lcp, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_max, blocks * sizeof(int)));
    
    // CPU -> GPU
    // lcp (CPU) viene copiato in d_lcp (GPU)
    CUDA_CHECK(cudaMemcpy(d_lcp, lcp, n * sizeof(int), cudaMemcpyHostToDevice));
    
    // Lancio del kernel di riduzione parallela
    reduce_max_kernel<<<blocks, threads, threads * sizeof(int)>>>(d_lcp, d_partial_max, n);
    
    // Trasferimento risultato parziale GPU -> CPU 
    // d_partial_max (GPU) viene copiato in h_partial_max (CPU)
    int *h_partial_max = (int*)malloc(blocks * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_partial_max, d_partial_max, blocks * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Sequenziale: si scorre l'array h_partial_max per trovare il massimo finale
    int max_lcp = 0;
    for(int i = 0; i < blocks; i++) {
        if(h_partial_max[i] > max_lcp) max_lcp = h_partial_max[i];
    }
    
    free(h_partial_max);
    CUDA_CHECK(cudaFree(d_lcp));
    CUDA_CHECK(cudaFree(d_partial_max));
    return max_lcp;
}
