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

#ifndef SEQ_SUFFIX_ARRAY_H
#define SEQ_SUFFIX_ARRAY_H
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Struttura che rappresenta un suffisso durante l'ordinamento.
 * - index: contiene l'indice di partenza del suffisso all'interno del testo.
 * - rank[2]:
 *   - rank[0]: rappresenta il rank del blocco corrente. Esso contiene il rank
 *     lessicografico del blocco di lunghezza h che inizia a i.
 *   - rank[1]: rappresenta il rank del blocco successivo. Esso contiene il rank
 *     lessicografico del blocco di lunghezza h che inizia a i+h.
 */
typedef struct {
    int index;      
    int rank[2];    
} Suffix;

/* FUNZIONE: suffixSort
 * Scopo: è la funzione primaria che costruisce il Suffix Array (SA).
 * Algoritmo Implementato: utilizza l'algoritmo di Manber-Myers.
 * Parametri:
   - text: che è il testo di input, rappresentato come un array di interi.
   - sa: è il Suffix Array da popolare. Sarà un array di interi di dimensione n.
   - n: è la lunghezza del testo.
 * Complessità Temporale: O(n log n)
 * Risultato: il Suffix Array sa conterrà gli indici di partenza dei suffissi del
 * testo, ordinati lessicograficamente.
*/
void suffixSort(int *text, int *sa, int n);

/* FUNZIONE: getHeight
 * Scopo: calcola il Longest Common Prefix Array. LCP[i] è la lunghezza del 
 * Prefisso Comune Più Lungo tra il suffisso all'indice sa[i-1] e il suffisso 
 * all'indice sa[i] del Suffix Array ordinato.
 * Algoritmo Implementato: utilizza l'algoritmo di Kasai.
 * Parametri:
 * - text: è il testo di input.
 * - sa: è il Suffix Array già calcolato da suffixSort.
 * - lcp: è l'LCP Array da popolare. Sarà un array di interi di dimensione n.
 * - n: è la lunghezza del testo.
 * Complessità Temporale: O(n)
 * Risultato: l'LCP Array lcp conterrà le lunghezze dei prefissi comuni più
 * lunghi tra suffissi adiacenti nel Suffix Array.
*/
void getHeight(int *text, int *sa, int *lcp, int n);

/* FUNZIONE: findLRS
 * Scopo: trova la lunghezza della Longest Repeated Substring, ovvero la 
 * sottostringa più lunga che appare almeno due volte nel testo.
 * Algoritmo Implementato: la lunghezza della LRS è semplicemente il valore 
 * massimo contenuto nell'LCP Array.
 * Parametri:
 * - lcp: è l'LCP Array già calcolato da getHeight.
 * - n: è la lunghezza del testo.
 * Complessità Temporale: O(n)
 * Risultato: restituisce un intero che è la lunghezza della LRS.
 */
int findLRS(int *lcp, int n);

#ifdef __cplusplus
}
#endif

#endif