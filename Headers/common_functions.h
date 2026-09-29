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

#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H
#include <stdlib.h> 

#ifdef __cplusplus
extern "C" {
#endif

// Ritorna il tempo di esecuzione in secondi.
double getTime(void);

/* FUNZIONE: readBinaryFile
 * Legge un file binario e lo converte in un array di interi.
 * - *out_text: puntatore all'array di interi allocato.
 * - *out_size: dimensione del file in byte.
 * RITORNA: 0 per successo, -1 per errore.
 */
int readBinaryFile(const char *filename, int **out_text, long *out_size);

#ifdef __cplusplus
}
#endif

#endif