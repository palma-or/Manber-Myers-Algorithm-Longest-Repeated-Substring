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

#include "merge_algorithms.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* FUNZIONE: compare_suffixes
 * Questa funzione è stata definita static inline:
 * - static: perchè può essere chiamata solo all'interno del file sorgente in cui 
 *   è definita. Ma soprattutto per garantire che non vengano commessi errori visto
 *   che è stata definita una funzione con la stessa intestazione nella versione sequenziale
 *   che non è static.
 * - inline: per suggerire al compilatore di eseguire l'espansione in linea. In questo modo
 *   si elimina l'overhead di chiamata, che è importante perchè la funzione viene chiamata nel
 *   loop, ottenendo un aumento della velocità
 * 
 * All'inizio del ciclo while, vengono eseguite tre azioni:
 * 1. Viene letto il carattere del primo suffisso e assegnato a c1: c1 = text[idx1 + pos]
 * 2. Viene letto il carattere del secondo suffisso e assegnato a c2: c2 = text[idx2 + pos]
 * 3. Viene verificata la condizione: (c1 == c2) && (c1 >= 0)
 * Il ciclo continua solo se i caratteri sono uguali (c1 == c2), quindi, non avendo 
 * trovato differenze, il confronto deve proseguire e se il carattere non è il 
 * terminatore (c1 >= 0). Se c1 fosse negativo (terminatore), il loop si 
 * interromperebbe, indicando che i suffissi sono identici fino alla fine del testo.
 * Se i suffissi matchano, si incrementa la posizione per il confronto del carattere successivo.
 * RISULTATO:
 * 1. Suffisso 1 è Maggiore --> S_idx1 > S_idx2
 *  - c1 > c2 è vero --> 1
 *  - c1 < c2 è falso --> 0
 *  - return 1 - 0 = 1 --> L'array di merge copierà l'elemento S_idx2 prima di S_idx1
 * 2. Suffisso 1 è Minore --> S_idx1 < S_idx2
 *  - c1 > c2 è falso --> 0
 *  - c1 < c2 è vero --> 1
 *  - return 0 - 1 = -1 --> L'array di merge copierà l'elemento S_idx1 prima di S_idx2
 * 3. Suffissi Identici --> S_idx1 == S_idx2
 *  - c1 > c2 è falso --> 0
 *  - c1 < c2 è falso --> 0
 *  - return 0 - 0 = 0 --> I suffissi sono considerati equivalenti
 */
static inline int compare_suffixes(int *text, int idx1, int idx2) {
    int pos = 0;
    int c1, c2;

    while ((c1 = text[idx1 + pos]) == (c2 = text[idx2 + pos]) && c1 >= 0) {
        pos++;
    }
    return (c1 > c2) - (c1 < c2);
}

/* FUNZIONE: merge_two_suffix_arrays
 * Questa funzione implementa l'algoritmo di Merge Sort applicato a due Suffix Array
 * già ordinati con lo scopo di fonderli in un unico SA globale ordinato.
 * COMPLESSITÀ: O(n1 + n2), dove n1 = size1, n2 = size2
 * PARAMETRI:
 * - *text: è il testo completjo globale. Serve per avere accesso a tutti i caratteri, 
 *   anche quelli fuori dai chunk originali
 * - *sa1: è il primo Suffix Array già ordinato e contiene gli indici globali
 * - *sa2: è il secondo Suffix Array già ordinato e contiene gli indici globali
 * - size1: è la dimensione del primo Suffix Array
 * - size2: è la dimensione del secondo Suffix Array
 * - *result: è l'array di outout, allocato per contenere tutti gli elementi di sa1 e sa2.
 *   Infatti, ha dimensione size1 + size2
 */
static void merge_two_suffix_arrays(int *text, int *sa1, int size1, int *sa2, int size2, int *result){
    int i = 0;  // Indice per scorrere sa1
    int j = 0;  // Indice per scorrere sa2
    int k = 0;  // Indice per scrivere in result
    
    /* 1. MERGE PRINCIPALE
     * Vengono scorsi entrambi gli array fino alla fine. Inoltre, vengono estratti gli
     * indici globali dei due suffissi che si trovano attualmente in testa a ciascuna lista 
     * ordinata e viene chiamata la funzione compare_suffixes per confrontare il suffisso
     * che inizia in idx1 con quello che inizia in idx2. cmp avrà quindi valore 0, 1 o -1.
     * - Se cmp è -1 o 0 --> S_idx1 è minore o uguale a S_idx2 --> Si sceglie l'elemento da sa1.
     *   L'indice sa1[i] viene copiato in result[k] e vengono incrementati gli indici i e k. 
     * - Se cmp è 1 --> S_idx2 è strettamente minore di S_idx1 --> Si sceglie l'elemento da sa2. 
     *   L'indice sa2[j] viene copiato in result[k] e vengono incrementati gli indici j e k.
    */
    while (i < size1 && j < size2) {
        
        int idx1 = sa1[i];
        int idx2 = sa2[j];
        
        int cmp = compare_suffixes(text, idx1, idx2);
        
        if (cmp <= 0) {
            result[k++] = sa1[i++];
        } else {
            result[k++] = sa2[j++];
        }
    }
    
    /* 2. COPIA ELEMENTI RESTANTI
     * Quando la Fase 1 termina, uno dei due array è arrivato alla fine ma, se ci sono elementi 
     * restanti nell'altro array, questi vengono semplicemente copiati in sequenza nell'array result. 
     * Poiché l'array restante era già ordinato, l'ordine finale di result è mantenuto.
    */
    while (i < size1) {
        result[k++] = sa1[i++];
    }
    
    while (j < size2) {
        result[k++] = sa2[j++];
    }
}

/* FUNZIONE: iterative_merge
 * Questa funzione implementa una fusione logaritmica per combinare tutti i Suffix Array
 * parziali. L'approccio consiste nel prender i Suffix Array parziali, uno per ogni 
 * processo MPI, e fonderli progressivamente in un Suffix Array Globale.
 * PARAMETRI:
 * - *text: è il testo completo globale
 * - **partial_sa: è l'array di puntatori che contiene tutti i Suffix Array parziali 
 *   inviati dai processi MPI. 
 * - *sizes: è l'array che contiene le dimensioni esatte di ciascun Suffix Array 
 *   parziale. Corrisponde a recvcounts in mpi_suffix_array.c.
 * * - num_chunks: è il numero totale di Suffix Array parziali da fondere.
 * * - total_n: è la dimensione totale del testo. Usata per l'allocazione 
 *   del risultato finale e per le verifiche di integrità.
 * * - *result: è il Suffix Array globale finale, completamente ordinato. 
 * 
 * STRATEGIA:
 * - Round 1: Fondo [SA0+SA1], [SA2+SA3], [SA4+SA5], ...
 * - Round 2: Fondo [SA01+SA23], [SA45+SA67], ...
 * - Round 3: Fondo [SA0123+SA4567], ...
 * - Continuo fino ad avere un unico SA globale
 */
void iterative_merge(int *text, int **partial_sa, int *sizes, int num_chunks, long total_n, int *result){

    /* 1. VERIFICA E INIZIALIZZAZIONE
     * Si esegue una verifica della dimensione calcolando total_check che è la somma delle 
     * dimensioni di tutti i chunk, per verificare che sia uguale alla dimensione totale 
     * attesa total_n. 
     * Ciò viene fatto per assicurarsi che non siano stati persi o duplicati byte durante
    */
    long total_check = 0;
    for (int i = 0; i < num_chunks; i++) {
        total_check += sizes[i];
    }
    
    if (total_check != total_n) {
        fprintf(stderr, "WARNING: Somma sizes (%ld) ≠ total_n (%ld)\n", 
                total_check, total_n);
    }
    
    /* 2. ALLOCAZIONE BUFFER TEMPORANEI
     * Vengono allocati due array di supporto di dimensione num_chunks:
     * - **buffers: è un array di puntatori ed è il registro che, di round in round, 
     *   memorizza i puntatori ai Suffix Array ancora attivi.
     * - *current_sizes: è un array che traccia la dimensione corrente di ogni Suffix 
     *   Array puntato da buffers.
    */
    int **buffers = (int **)malloc(num_chunks * sizeof(int *));
    int *current_sizes = (int *)malloc(num_chunks * sizeof(int));
    
    if (!buffers || !current_sizes) {
        fprintf(stderr, "ERRORE: Impossibile allocare buffer per merge\n");
        return;
    }
    
    // Inizialmente, i buffer puntano ai SA parziali originali
    for (int i = 0; i < num_chunks; i++) {
        buffers[i] = partial_sa[i];
        current_sizes[i] = sizes[i];
    }
    
    int active_chunks = num_chunks; 
    int round = 1;
    
    /* 3. MERGE ITERATIVO
     * Questo loop continua a fondere array con l'obiettivo di dimezzare il numero di
     * array attivi finchè non ne resta solo uno.
     * - active_chunks: è il numero di Suffix Array separati che devono ancora essere 
     *   fusi. 
     * - round: è il contatore che traccia il round di merge
    */
    while (active_chunks > 1) {
                
        int new_active = 0; 
        
        for (int i = 0; i < active_chunks; i += 2) {
            
            if (i + 1 < active_chunks) {

                // CHUNK PARI --> Viene calcolata la dimensione del merge e allocato un nuovo buffer
                int size_merged = current_sizes[i] + current_sizes[i+1];
                int *merged = (int *)malloc(size_merged * sizeof(int));
                
                if (!merged) {
                    fprintf(stderr, "ERRORE: Impossibile allocare memoria per merge\n");
                    fprintf(stderr, "Richiesti: %d bytes\n", size_merged * (int)sizeof(int));
                    for (int j = 0; j < new_active; j++) {
                        if (round > 1) free(buffers[j]);
                    }
                    free(buffers);
                    free(current_sizes);
                    return;
                }
                                
                // Si esegue il merge
                merge_two_suffix_arrays(text, buffers[i], current_sizes[i], buffers[i+1], current_sizes[i+1], merged);
                
                // buffers[i] e buffers[i+1] sono stati fusi, quindi non sono più necessari
                // round = 1 si ha partial_sa che deve essere liberato solo alla fine
                if (round > 1) {
                    free(buffers[i]);
                    free(buffers[i+1]);
                }
                
                // Salva il risultato del merge
                buffers[new_active] = merged;
                current_sizes[new_active] = size_merged;
                new_active++;
                
            } else {

                /* CHUNCH DISPARI --> Se active_chunks è dispari, l'ultimo Suffix Array 
                 * non ha un partner, quindi, non si esegue il merge e il puntatore del
                 * chunk e la sua dimensione vengono copiati nella lista dei nuovi 
                 * chunk attivi. In questo modo verrà fuso nel round successivo con il 
                 * risultato del merge precedente.
                */
                buffers[new_active] = buffers[i];
                current_sizes[new_active] = current_sizes[i];
                new_active++;
            }
        }
        
        active_chunks = new_active;
        round++;
    }
    
    // 4. buffers[0] viene copiato in result, fornito da Rank 0
    
    memcpy(result, buffers[0], total_n * sizeof(int));
    
    // Si fa il controllo per assicurarsi che sia iterative_merge che possiede questa memoria
    // Infatti, round = 1 è la funzione chiamante a possedere il puntatore 
    if (round > 1) {
        free(buffers[0]);
    }
    free(buffers);
    free(current_sizes);
}