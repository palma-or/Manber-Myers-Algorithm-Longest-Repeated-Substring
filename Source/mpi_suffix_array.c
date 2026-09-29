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

#include "mpi_suffix_array.h"
#include "seq_suffix_array.h"
#include "merge_algorithms.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* FUNZIONE: read_and_sort_chunk
 * Questa funzione è eseguita in parallelo da ogni processo MPI (identificato da rank) 
 * e ha il compito di:
 * - Determinare in modo equo la porzione di file da leggere.
 * - Leggere questa porzione usando l'I/O parallelo.
 * - Calcolare il Suffix Array (SA) locale su questa porzione.
 * - Correggere gli indici locali in indici globali, preparando il 
 *   risultato per la fase di merge.
 */
void read_and_sort_chunk(const char *filename, int rank, int size, int **local_text, int **local_sa, long *out_chunk_size, long *out_total_size, MPI_Offset *out_offset){
    
    /* 1. APERTURA FILE PARALLELA 
     * Si utilizza MPI_File_open per aprire il file binario, garantendo che 
     * tutti i processi abbiano accesso al file system in modo coordinato.
     * Viene aperto il file in modalità di sola lettura (MPI_MODE_RDONLY) 
     * su tutti i processi che fanno parte del comunicatore globale (MPI_COMM_WORLD). 
     * fh è il file handle che identifica il file aperto.
    */
    MPI_File fh;
    int err = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    
    if (err != MPI_SUCCESS) {
        if (rank == 0) {
            fprintf(stderr, "ERRORE CRITICO: Impossibile aprire il file '%s'\n", filename);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* 2.CALCOLO DIMENSIONI
     * MPI_File_get_size è chiamata per ottenere la dimensione totale del 
     * file in byte (total_len)
    */
    MPI_Offset total_len;
    MPI_File_get_size(fh, &total_len);
    *out_total_size = (long)total_len;

    /* 3. DISTRIBUZIONE DEL LAVORO
     * Questa è la logica che assicura che il carico sia bilanciato tra tutti i processi.
     * - base_chunk è il numero minimo di byte che ogni processo riceverebbe 
     *   se il file fosse perfettamente divisibile.
     * - remainder è il resto della divisione. Se il resto è R, significa che i 
     *   primi R processi, da rank=0 a rank=R-1, devono ricevere un byte extra.
    */
    long base_chunk = *out_total_size / size;
    long remainder = *out_total_size % size;

    /* - *out_chunk_size: si assegna il base_chunk più un byte extra se il 
     *   rank corrente è inferiore a remainder.
     * - *out_offset: è l'indice di partenza nel file globale e si calcola come:
     *   rank * base_chunk, che rappresenta la porzione base per tutti i processi.
     *   a cui vengono aggiunti i byte extra che i processi precedenti al corrente hanno 
     *   ricevuto (rank se rank < remainder, altrimenti si aggiunge il resto remainder).
     * Risultato: Ogni processo ora conosce esattamente dove iniziare a leggere (*out_offset)
     * e quanti byte leggere (*out_chunk_size).
    */
    *out_chunk_size = base_chunk + (rank < remainder ? 1 : 0);
    *out_offset = rank * base_chunk + (rank < remainder ? rank : remainder);
    
    
    unsigned char *buffer = (unsigned char*)malloc(*out_chunk_size);
    if (!buffer) {
        fprintf(stderr, "[Rank %d] Errore allocazione buffer\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* 4. LETTURA PARALLELA
     * MPI_File_read_at_all viene utilizzata per consentire ad ogni rank di leggere
     * direttamente la sua porzione di file dal disco, partendo dal suo *out_offset.
     * L'alternativa sarebbe stata quella di far leggere tutto al rank 0 e poi
     * distribuire la porzione di file assegnata a ciascun processo. Ciò però
     * avrebbe creato un collo di bottiglia e rallentato l'intero processo.
     * Parametri:
     * - fh: file handle del file aperto.
     * - *out_offset: offset nel file da cui iniziare la lettura per questo processo.
     * - buffer: buffer locale dove i dati letti saranno memorizzati.
     * - *out_chunk_size: numero di byte da leggere.
     * - MPI_BYTE: tipo di dato MPI che indica che si stanno leggendo byte grezzi.
    */
    MPI_File_read_at_all(fh, *out_offset, buffer, (int)(*out_chunk_size), MPI_BYTE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    /* 5. CONVERSIONE 
     * I byte letti nel buffer temporaneo (unsigned char*) vengono iterati e 
     * convertiti nel formato finale (int*) nell'array *local_text. 
     * Questo formato è necessario per l'algoritmo di Manber-Myers, che 
     * richiede che il testo sia rappresentato da un array di interi per poter 
     * manipolare i rank e i valori dei caratteri in modo uniforme e robusto. 
     * Quindi, viene allocato dinamicamente lo spazio sufficiente per memorizzare 
     * il chunk di testo letto in un array di interi.
     * La dimensione allocata è (*out_chunk_size) * sizeof(int). 
     * Dato che un int occupa tipicamente più byte di un char, questo array 
     * sarà più grande del buffer temporaneo in termini di spazio di memoria 
     * totale, ma conterrà lo stesso numero di elementi (*out_chunk_size).
    */
    *local_text = (int*)malloc((*out_chunk_size) * sizeof(int));
    if (!*local_text) {
        fprintf(stderr, "[Rank %d] Errore allocazione local_text\n", rank);
        free(buffer);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (long i = 0; i < *out_chunk_size; i++) {
        (*local_text)[i] = (int)buffer[i];
    }
    free(buffer);

    *local_sa = (int*)malloc((*out_chunk_size) * sizeof(int));
    if (!*local_sa) {
        fprintf(stderr, "[Rank %d] Errore allocazione local_sa\n", rank);
        free(*local_text);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* 6. CALCOLO SUFFIX ARRAY LOCALE
     * Il processo lancia la funzione sequenziale suffixSort sul testo locale 
     * (*local_text) di dimensione *out_chunk_size.
     * Risultato: *local_sa contiene ora gli indici di partenza dei suffissi, 
     * ordinati lessicograficamente, relativamente solo al chunk locale. 
    */
    suffixSort(*local_text, *local_sa, (int)(*out_chunk_size));
    
    /* 7. Globalizzazione indici
     * Prima di inviare il Suffix Array alla fase di merge, gli indici locali devono
     * essere corretti per riflettere la loro vera posizione nel testo globale. quindi, 
     * ad ogni indice (*local_sa)[i], che è un valore locale da 0 a chunk_size - 1, 
     * viene aggiunto l'offset globale di partenza (*out_offset).    
    */
    for (long i = 0; i < *out_chunk_size; i++) {
        (*local_sa)[i] += (int)(*out_offset);
    }
}

/* FUNZIONE: gather_and_merge
 * Questa funzione è responsabile della transizione dai Suffix Array Locali al 
 * Suffix Array Globale finale. Gran parte del lavoro avviene sul Rank 0, mentre 
 * gli altri processi partecipano inviando i loro dati.
 * La funzione utilizza MPI_Gather per raccogliere le dimensioni dei chunk da 
 * ogni processo e MPI_Gatherv per raccogliere il testo completo. 
 * Tuttavia, per i Suffix Array, si adotta un approccio diverso. Infatti, ogni 
 * processo invia il proprio Suffix Array al Rank 0 tramite MPI_Send, che li 
 * memorizza separatamente in un array di puntatori. In questo modo si mantengono 
 * i Suffix Array parziali separati, facilitando un merge ottimizzato successivo.
 * Una volta raccolti tutti i Suffix Array parziali, il Rank 0 invoca la funzione 
 * iterative_merge, che implementa un algoritmo di merge efficiente per combinare i 
 * Suffix Array parziali in un unico Suffix Array globale ordinato.
 * Parametri:
 * - local_text: Puntatore al testo locale del processo corrente.
 * - local_sa: Puntatore al Suffix Array locale del processo corrente.
 * - chunk_size: Dimensione del chunk locale (numero di caratteri).
 * - rank: Identificatore del processo MPI corrente.
 * - size: Numero totale di processi MPI.
 * - total_n: Dimensione totale del testo (numero di caratteri).
 * - global_text: Puntatore al puntatore dove memorizzare il testo globale (solo sul Rank 0).
 * - global_sa: Puntatore al puntatore dove memorizzare il Suffix Array globale (solo sul Rank 0).
*/
void gather_and_merge(int *local_text, int *local_sa, long chunk_size, int rank, int size, long total_n, int **global_text, int **global_sa) 
{
    int *recvcounts = NULL;
    int *displs = NULL;
    
    /* 1. Preparazione della Memoria Globale (solo Rank 0)
     * Vengono allocati due array della dimensione totale del testo (total_n):
     * - *global_text: contiene l'intero testo, necessario per i confronti lessicografici 
     *   nella fase di merge.
     * - *global_sa: contiene il Suffix Array Globale finale, il risultato cercato.
     * Inoltre, vengono allocati due array di supporto:
     * - recvcounts: memorizza la dimensione di ciascun chunk inviato da ogni processo.
     * - displs: memorizza gli offset di partenza per ciascun chunk nel testo globale.
    */
    if (rank == 0) {
        
        *global_text = (int*)malloc(total_n * sizeof(int));
        *global_sa = (int*)malloc(total_n * sizeof(int));
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        if (!*global_text || !*global_sa || !recvcounts || !displs) {
            fprintf(stderr, "[Rank 0] ERRORE: Allocazione memoria per gather\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* 2. GATHER DELLE DIMENSIONI DEI CHUNK
     * Tutti i processi inviano la dimensione del loro chunk (my_size) al Rank 0. 
     * Il Rank 0 li memorizza nell'array recvcounts.
     * Solo il Rank 0 procede a calcolare l'array displs.
     * Per i Rank successivi, displs[i] è la somma delle dimensioni dei chunk dei 
     * processi precedenti (0 a i-1). Questo garantisce che i dati del Rank i vengano 
     * scritti in *global_text subito dopo la fine dei dati del Rank i-1.
    */
    int my_size = (int)chunk_size;
    MPI_Gather(&my_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }

    /* 3. GATHER DEL TESTO GLOBALE
     * Viene utilizzata MPI_Gatherv per raccogliere i chunk di testo da tutti i processi 
     * e assemblarli in *global_text sul Rank 0, gestendo automaticamente le dimensioni 
     * variabili dei chunk grazie ai vettori recvcounts e displs.
     * MPI_Gatherv è necessaria qui perché i chunk possono avere dimensioni diverse.
     * Inoltre, avere l'intero testo in *global_text sul Rank 0 è fondamentale perchè il 
     * merge deve confrontare suffissi che iniziano in chunk diversi e si estendono 
     * oltre i confini dei chunk, quindi, è necessario poter accedere a tutti i caratteri.
    */
    
    MPI_Gatherv(local_text, my_size, MPI_INT, *global_text, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* 4. GATHER DEI SUFFIX ARRAY PARZIALI E MERGE
     * Invece di usare MPI_Gatherv per raccogliere i Suffix Array parziali, 
     * ogni processo invia il proprio Suffix Array al Rank 0 tramite MPI_Send.
     * MPI_Gatherv non viene utilizzato perchè i Suffix Array non possono essere
     * semplicemente concatenati come il testo. Ogni Suffix Array rappresenta
     * indici che devono essere mantenuti come liste separate, altrimenti si perderebbe
     * l'ordinamento corretto necessario per il merge (non ho la priority queue)
     * Il Rank 0 memorizza ciascun Suffix Array in un array di puntatori partial_sa.
     * Una volta raccolti tutti i Suffix Array parziali, il Rank 0 invoca la 
     * funzione iterative_merge, che implementa un algoritmo di merge efficiente 
     * per combinare i Suffix Array parziali in un unico Suffix Array globale ordinato.
    */
    
    if (rank == 0) {
        
        // Alloca l'array di puntatori per i SA parziali
        int **partial_sa = (int **)malloc(size * sizeof(int *));
        if (!partial_sa) {
            fprintf(stderr, "Errore allocazione partial_sa\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Alloca spazio per ciascun SA parziale
        for (int i = 0; i < size; i++) {
            partial_sa[i] = (int *)malloc(recvcounts[i] * sizeof(int));
            if (!partial_sa[i]) {
                fprintf(stderr, "Errore allocazione partial_sa[%d]\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        
        // Ricevi i SA parziali uno per uno
        for (int i = 0; i < size; i++) {
            if (i == 0) {
                // memcpy copia il local_sa di Rank 0 nell'array partial_sa[0].
                memcpy(partial_sa[0], local_sa, recvcounts[0] * sizeof(int));
            } else {
                // MPI_Recv riceve il Suffix Array inviato dal Rank i e lo memorizza in partial_sa[i].
                MPI_Recv(partial_sa[i], recvcounts[i], MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        /* La funzione iterative_merge prende i Suffix Array parziali, già ordinati e 
         * con indici globalizzati e li fonde in un unico array ordinato.
         * Il merge procede scegliendo, ad ogni passo, il suffisso lessicograficamente 
         * minimo tra i suffissi puntati dai puntatori di merge. Il confronto lessicografico 
         * completo viene eseguito utilizzando l'array *global_text, assicurando la correttezza 
         * anche per i suffissi che attraversano i confini dei chunk.
         * Risultato: Il Suffix Array Globale corretto viene scritto in *global_sa.
        */
        iterative_merge(*global_text, partial_sa, recvcounts, size, total_n, *global_sa);
        
        for (int i = 0; i < size; i++) {
            free(partial_sa[i]);
        }
        free(partial_sa);
        free(recvcounts);
        free(displs);
        
    } else {
        /* I processi con rank > 0 inviano semplicemente il loro Suffix Array Locale
         * al rank 0 usando MPI_Send.
        */
        MPI_Send(local_sa, my_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    
    /* La barriera garantisce che tutti i processi attendano che il Rank 0 abbia completato 
     * il merge e la pulizia della memoria prima di terminare o passare alla fase successiva  
    */
    MPI_Barrier(MPI_COMM_WORLD);
}