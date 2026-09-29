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
#include "cuda_suffix_array.h"
#include "seq_suffix_array.h"
#include "common_functions.h"


int main(int argc, char *argv[]) {
    printf("SUFFIX ARRAY - VERSIONE CUDA \n");
    
    if (argc != 2) {
        fprintf(stderr, "ERRORE: Numero di argomenti non valido\n\n");
        return 1;
    }
    
    /* VARIABILI PRINCIPALI:
     * - *text: è l'array che conterrà il testo letto dal file
     * - *sa: è l'array che conterrà il Suffix Array
     * - *lcp: è l'array che conterrà l'LCP Array
     * - n: è la dimensione del testo in byte
    */
    int *text = NULL;       
    int *sa = NULL;         
    int *lcp = NULL;        
    long n = 0;             
    
    // 1. LETTURA DEL FILE tramite la funzione readBinaryFile
    if (readBinaryFile(argv[1], &text, &n) != 0) {
        fprintf(stderr, "Errore nella lettura del file.\n");
        return 1;
    }

    // Conversione long a int per le funzioni di Suffix Array
    int n_int = (int)n;

    // Allocazione Suffix Array
    sa = (int*)malloc(n * sizeof(int));
    if (!sa) {
        fprintf(stderr, "Errore: Impossibile allocare Suffix Array\n");
        free(text);
        return 1;
    }
    
    // Allocazione LCP Array
    lcp = (int*)malloc(n * sizeof(int));
    if (!lcp) {
        fprintf(stderr, "Errore: Impossibile allocare LCP Array\n");
        free(text);
        free(sa);
        return 1;
    }
    
    // 2. COSTRUZIONE SUFFIX ARRAY (CPU)
    double start_sa = getTime();
    suffixSort(text, sa, n_int);
    double time_sa = getTime() - start_sa;

    // 3. COSTRUZIONE LCP ARRAY (CPU)
    double start_lcp = getTime();
    getHeight(text, sa, lcp, n_int);
    double time_lcp = getTime() - start_lcp;

    // 4. RICERCA LRS (CUDA)
    double start_lrs = getTime();
    int lrs = findLRSCUDA(lcp, n_int);
    double time_lrs = getTime() - start_lrs;

    // STAMPA DEI RISULTATI FINALI
    double total_time = time_sa + time_lcp + time_lrs;
    printf("------- RISULTATI ------- \n");
    printf("   File:                     %s\n", argv[1]);
    printf("   Dimensione:               %ld bytes (%.2f MB)\n", n, n / 1024.0 / 1024.0);
    printf("   LRS:                      %d caratteri\n\n", lrs);
    
    printf("TEMPI DI ESECUZIONE:\n");
    printf("   Suffix Array (CUDA):      %.4f s\n", time_sa);
    printf("   LCP Array (CUDA/CPU):     %.4f s\n", time_lcp);
    printf("   LRS Search (CUDA):        %.4f s\n", time_lrs);
    printf("   --------------------------------------------------\n");
    printf("   Totale:                   %.4f s\n\n", total_time);
    
    free(text);
    free(sa);
    free(lcp);
    
    return 0;
}