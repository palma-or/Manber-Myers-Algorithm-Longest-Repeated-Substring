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

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "seq_suffix_array.h"
#include "mpi_suffix_array.h"

/* FUNZIONE: get_time
 * Questa funzione utilizza MPI_Wtime(), che restituisce il tempo wall clock 
 * (tempo reale trascorso) con alta precisione. 
*/
double get_time() { 
    return MPI_Wtime(); 
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "ERRORE: Uso corretto: %s <file_binario>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("\n");
        printf("SUFFIX ARRAY - VERSIONE MPI \n");
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = get_time();

    // FASE 1: LETTURA E SORT LOCALE

    int *local_text = NULL;
    int *local_sa = NULL;
    long chunk_size, total_n;
    MPI_Offset my_offset;
    
    read_and_sort_chunk(argv[1], rank, size, &local_text, &local_sa, &chunk_size, &total_n, &my_offset);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_local_end = get_time();
    
    // FASE 2: GATHER E MERGE
    int *global_text = NULL;
    int *global_sa = NULL;
    
    gather_and_merge(local_text, local_sa, chunk_size, rank, size, total_n, &global_text, &global_sa);
    
    free(local_text);
    free(local_sa);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_merge_end = get_time();
    
    // FASE 3: LCP E LRS
    if (rank == 0) {
        
        int *lcp = (int*)malloc(total_n * sizeof(int));
        if (!lcp) {
            fprintf(stderr, " Errore allocazione LCP array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        getHeight(global_text, global_sa, lcp, (int)total_n);
        
        int lrs = findLRS(lcp, (int)total_n);
        
        double t_end = get_time();
        
        // STAMPE RISULTATI FINALI
        printf("------- RISULTATI -------\n");
        
        printf("INFORMAZIONI FILE:\n");
        printf("  File:                     %s\n", argv[1]);
        printf("  Dimensione:               %ld bytes\n", total_n);
        printf("  LRS (caratteri):          %d caratteri\n", lrs);
        printf("  Processi MPI:             %d\n\n", size);
        
        printf("TEMPI DI ESECUZIONE:\n");
        printf(" Lettura e Sort:      %.4f s  [%.1f%%]\n", t_local_end - t_start, 100.0 * (t_local_end - t_start) / (t_end - t_start));
        printf(" Gather e Merge:      %.4f s  [%.1f%%]\n", t_merge_end - t_local_end, 100.0 * (t_merge_end - t_local_end) / (t_end - t_start));
        printf(" LCP e LRS:           %.4f s  [%.1f%%]\n", t_end - t_merge_end, 100.0 * (t_end - t_merge_end) / (t_end - t_start));
        printf(" ----------------------------------------------------\n");
        printf(" TOTALE:              %.4f s\n\n", t_end - t_start);
        
        
        free(global_text);
        free(global_sa);
        free(lcp); 
    }

    MPI_Finalize();
    return 0;
}