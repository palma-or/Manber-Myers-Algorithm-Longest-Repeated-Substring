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

#ifndef MPI_SUFFIX_ARRAY_H
#define MPI_SUFFIX_ARRAY_H
#include <mpi.h>

/* FUNZIONE: read_and_sort_chunk
 * Gestisce la fase parallela iniziale: ogni processo MPI legge 
 * una porzione del file e calcola il Suffix Array LOCALE.
 * PARAMETRI:
 * - filename: nome del file binario di input
 * - rank: identifica il processo corrente
 * - size: identifica il numero totale di processi.
 * - *local_text: puntatore all'array locale di testo allocato e riempito.
 * - *local_sa: puntatore all'array locale di Suffix Array allocato e riempito.
 *   chiamante i puntatori agli array allocati e riempiti localmente.
 * - *out_offset: L'indice di partenza del chunk nel file globale. 
 * - *out_chunk_size: La dimensione del chunk assegnato a questo processo.
 * - *out_total_size: La dimensione totale del file (uguale per tutti i
 *   processi).
 */
void read_and_sort_chunk(const char *filename, int rank, int size, int **local_text, int **local_sa, long *out_chunk_size, long *out_total_size, MPI_Offset *out_offset);

/* FUNZIONE: gather_and_merge
 * Raccoglie tutti i risultati parziali su Rank 0 e costruisce il
 * Suffix Array GLOBALE usando il merge iterativo.
 * PARAMETRI:
 * - *local_text: array locale di testo (di dimensione chunk_size)
 * - *local_sa: array locale di Suffix Array (di dimensione chunk_size)
 * - chunk_size: dimensione del chunk locale
 * - rank: identifica il processo corrente
 * - size: identifica il numero totale di processi
 * - total_n: dimensione totale del testo (uguale per tutti i processi)
 * - *global_text: puntatore all'array globale di testo (solo su Rank 0)
 * - *global_sa: puntatore all'array globale di Suffix Array (solo su Rank 0) 
*/
void gather_and_merge(int *local_text, int *local_sa, long chunk_size, int rank, int size, long total_n, int **global_text, int **global_sa);

#endif