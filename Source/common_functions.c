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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "common_functions.h"

/* FUNZIONE: getTime
 * Scopo: Misurare il tempo impiegato dal processore nell'esecuzione del programma
 * per valutare l'efficienza degli algoritmi.
 * Utilizza la funzione standard clock() che restituisce il numero di tick di clock
 * trascorsi. Il risultato viene convertito in secondi dividendo per la costante 
 * CLOCKS_PER_SEC.  
 * RITORNA: Tempo in secondi
 */
double getTime(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

/* FUNZIONE: readBinaryFile
 * Legge file binari e prepara il testo in un formato adatto agli algoritmi.
 * Tenta di aprire il file specificato in modalità di lettura binaria ("rb") e, 
 * in caso di errore, stampa un messaggio e restituisce -1. 
 * Utilizza le funzioni fseek e ftell per posizionare il cursore alla fine del file 
 * e ottenere la sua dimensione esatta in byte (file_size). Successivamente,
 * alloca un buffer temporaneo di tipo unsigned char* (buffer) per leggere i byte 
 * grezzi dal file. Viene usato questo tipo perché ogni byte del file corrisponde 
 * a un carattere e unsigned char garantisce che il valore (0-255) sia letto 
 * correttamente.
 * Inoltre, alloca l'array di output int* (*out_text). Il testo viene memorizzato 
 * come un array di interi invece che come un array di caratteri. Perchè, 
 * sebbene gli algoritmi di Suffix Array possano lavorare con char, l'uso di int 
 * garantisce che l'algoritmo Manber-Myers possa gestire un alfabeto con caratteri
 * maggiori di 255.
 * La funzione fread legge l'intero contenuto del file nel buffer temporaneo. 
 * Viene effettuato un controllo per assicurarsi che il numero di byte letti sia 
 * uguale alla dimensione attesa. 
 * Infine, dopo aver chiuso il file e verificato la lettura, si converte ogni 
 * byte nel buffer (range 0-255) nel suo corrispondente intero nell'array finale 
 * ((*out_text)[i] = (int)buffer[i];).
 * 
 * -**out_text: è un puntatore a un puntatore. Questo permette alla funzione di scrivere
 *   l'indirizzo dell'array allocato (int*) nella variabile text del chiamante (main).
 * -*out_size: Permette di scrivere la dimensione del file (long) nella variabile n
 *   del chiamante.
 * RITORNA:
 * - 0 se successo
 * - -1 se errore
 */
int readBinaryFile(const char *filename, int **out_text, long *out_size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "ERRORE: Impossibile aprire il file '%s'\n", filename);
        return -1;
    }
    
    // DETERMINAZIONE DIMENSIONE FILE
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    *out_size = file_size;
    
    if (file_size == 0) {
        fprintf(stderr, "\n  WARNING: Il file è vuoto!\n");
        fclose(file);
        return -1;
    }
    
    // ALLOCAZIONE MEMORIA
    // Buffer temporaneo per leggere i byte
    unsigned char *buffer = (unsigned char*)malloc(file_size);
    if (!buffer) {
        fprintf(stderr, " ERRORE: Impossibile allocare %ld bytes per buffer\n", file_size);
        fclose(file);
        return -1;
    }
    
    // Array di interi per il testo
    *out_text = (int*)malloc(file_size * sizeof(int));
    if (!*out_text) {
        fprintf(stderr, " ERRORE: Impossibile allocare %ld bytes per testo\n", file_size * sizeof(int));
        free(buffer);
        fclose(file);
        return -1;
    }
    
    // LETTURA FILE
    size_t bytes_read = fread(buffer, 1, file_size, file);
    
    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, " ERRORE: Lettura incompleta\n");
        fprintf(stderr, "   Attesi:  %ld bytes\n", file_size);
        fprintf(stderr, "   Letti:   %zu bytes\n", bytes_read);
        free(buffer);
        free(*out_text);
        fclose(file);
        return -1;
    }

    fclose(file);
    
    // CONVERSIONE BYTE a INT
    for (long i = 0; i < file_size; i++) {
        (*out_text)[i] = (int)buffer[i];
    }
    
    free(buffer);    
    return 0;
}