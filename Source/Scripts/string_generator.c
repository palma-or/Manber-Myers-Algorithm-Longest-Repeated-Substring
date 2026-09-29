/* L'implementazione di xorshift64star è basata sul codice originale di Sebastiano Vigna.
 * Fonte: http://vigna.di.unimi.it/ftp/papers/xorshift.c
 *
 * Copyright originale dell'autore dell'algoritmo:
 * Written in 2014 by Sebastiano Vigna (vigna@acm.org)
 *
 * To the extent possible under law, the author has dedicated all copyright
 * and related and neighboring rights to this software to the public domain
 * worldwide. This software is distributed without any warranty.
 *
 * See <http://creativecommons.org/publicdomain/zero/1.0/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <direct.h>  

#define BUFFER_SIZE 1048576 // 1 MB = 1024 * 1024 = 1.048.576 byte
#define FIXED_SEED 12345678901234567ULL

/* xorshift_state: è un numero a 64 bit che viene modificato ad ogni chiamata.
 * Il suo valore determina quale sarà il prossimo numero generato. È dichiarato 
 * "static" perché deve mantenere il suo valore tra una chiamata e l'altra della 
 * funzione.
 * Non deve mai essere zero, altrimenti il generatore produrrebbe solo zeri 
 * (perché 0 XOR 0 = 0, 0 << n = 0, ecc.)
*/
static uint64_t xorshift_state = 1;

/* xorshift_seed: inizializza il generatore con un valore di partenza.
 * seed è il valore iniziale da cui partirà la sequenza pseudo-casuale.
*/
void xorshift_seed(uint64_t seed) {
    xorshift_state = seed;
    
    if (xorshift_state == 0) {
        xorshift_state = 1;
    }
}

/* xorshift64star: produce un numero a 64 bit.
 * XOR confronta due bit e restituisce:
 * - 1 se i bit sono diversi (0^1=1, 1^0=1)
 * - 0 se i bit sono uguali (0^0=0, 1^1=0)
 * Se si applica XOR due volte, si annulla l'effetto: A ^ B ^ B = A 
 * 
 * Funzionamento operatori di shift (>> e <<):
 * >> n : sposta tutti i bit verso destra di n posizioni (i bit a destra si perdono)
 * << n : sposta tutti i bit verso sinistra di n posizioni (i bit a sinistra si perdono)
 *
 * L'algoritmo xorshift: si combina XOR e shift in modo da mescolare i bit dello stato
 * in modo da ottenere una buona distribuzione di valori casuali.
 * 
 * I numeri 12, 25, 27 non sono casuali ma sono sono stati scelti matematicamente
 * per garantire che il generatore abbia il periodo massimo possibile (2^64-1)
 * e che passi i test statistici di casualità.
 * 
 * La moltiplicazione finale per una costante che vale 0x2545F4914F6CDD1D
 * migliora ulteriormente la qualità statistica dell'output. Questa variante
 * si chiama xorshift64* ed è stata proposta da Sebastiano Vigna.
*/
uint64_t xorshift64star(void) {
    uint64_t x = xorshift_state; // Prendiamo lo stato corrente
    
    /* 1. Si prende x e lo si sposta a destra di 12 bit per poi fare XOR col valore originale.*/
    x ^= x >> 12;
    
    /* 2. Si sposta a sinistra di 25 bit e si fa di nuovo XOR.*/
    x ^= x << 25;
    
    /* 3. Si effettua un ultimo shift a destra di 27 bit in modo da garantire 
     * che ogni bit dell'output dipenda da diversi bit dello stato iniziale.
    */
    x ^= x >> 27;
    
    // Si aggiorna lo stato con il nuovo valore di x
    xorshift_state = x;
    
    /* 4. Si moltiplica per una costante esadecimale 0x2545F4914F6CDD1D,
     * che è stata scelta perché produce una distribuzione statistica molto buona.
     * In decimale vale: 2685821657736338717
     * Il suffisso ULL indica "unsigned long long" (64 bit).
    */
    return x * 0x2545F4914F6CDD1DULL;
}


/* La funzione random_byte genera un singolo byte casuale (valore da 0 a 255).
 * Essa chiama xorshift64star() per ottenere un numero a 64 bit e poi
 * estrae gli 8 bit meno significativi usando l'operatore AND bit a bit (&). * 
 * Il cast a (unsigned char) garantisce che il risultato sia un byte (0-255).
*/
unsigned char random_byte(void) {
    return (unsigned char)(xorshift64star() & 0xFF);
}

/* Configurazione delle dimensioni target per i file di test.
 * Verranno generati file binari casuali di diverse dimensioni
 * per essere utilizzati nei test dell'algoritmo Manber-Myers.
*/
int TARGET_SIZES[] = {1, 50, 100, 200, 500};
int NUM_SIZES = 5;

/* Il fattore di overhead della memoria rappresenta il rapporto tra 
 * la memoria totale occupata e la dimensione effettiva della stringa. 
 * Viene utilizzato per calcolare quanti byte scrivere nel file, 
 * considerando l'overhead di memoria introdotto dalle strutture dati dell'algoritmo.
 * Infatti, la memoria totale è data dalla somma della memoria per la stringa e quella
 * utilizzata dalle strutture dati per il suffix array, per l'LCP array e ausiliarie.
*/
int OVERHEAD = 21;

int main() {

    _mkdir("Data");
    
    // Inizializzazione del generatore con un seed fisso
    xorshift_seed(FIXED_SEED);
        
    
    /* 2. Si genera un file per ogni dimensione target.
     * La variabile idx (index) scorre da 0 a NUM_SIZES-1, permettendo
     * di accedere a ogni elemento dell'array TARGET_SIZES.
    */
    for (int idx = 0; idx < NUM_SIZES; idx++) {
        
        // Si legge la dimensione target in megabyte per questa iterazione.
        int target_mb = TARGET_SIZES[idx];
        
        /* Viene calcolata la dimensione effettiva della stringa in byte
         * da scrivere nel file, tenendo conto dell'overhead di memoria.
         * Formula: str_size_bytes = (target_mb * 1024 * 1024) / OVERHEAD
         * - target_mb * 1024 * 1024 converte MB in byte (1 MB = 1024 KB = 1024*1024 byte)
         * - Si divide per OVERHEAD per ottenere la dimensione della stringa
         *   che, con le strutture dati aggiuntive, userà target_mb di memoria totale
         * 
         * Usiamo "long long" (almeno 64 bit) perché con target_mb = 500,
         * il prodotto 500 * 1024 * 1024 = 524.288.000 che sta in un int,
         * ma è buona pratica usare tipi più grandi per evitare overflow.
        */
        long long str_size_bytes = ((long long)target_mb * 1024 * 1024) / OVERHEAD;
        
        char filename[256];
        sprintf(filename, "Data/string_%dMB.bin", target_mb);
        
        // Apertura del file in modalità scrittura binaria
        FILE *file = fopen(filename, "wb");
        
        if (!file) {
            fprintf(stderr, "ERRORE: Impossibile creare il file %s\n", filename);
            continue;  
        }
        
        /* Allocazione del buffer di scrittura.
         * malloc (memory allocate) riserva un blocco di memoria della dimensione
         * specificata e restituisce un puntatore ad esso.
         * Scrivere sul disco è un'operazione lenta. Ogni chiamata a fwrite
         * potrebbe richiedere una "system call" al sistema operativo.
         * Se scrivessimo un byte alla volta, faremmo milioni di system call!
         * Invece, accumuliamo 1 MB di dati in memoria (nel buffer) e poi
         * li scriviamo tutti insieme con una sola chiamata a fwrite.
        */
        unsigned char *buffer = (unsigned char*)malloc(BUFFER_SIZE);
        
        if (!buffer) {
            fprintf(stderr, "ERRORE: Allocazione buffer fallita\n");
            fclose(file);  
            continue;
        }
        
        // Contatore di byte scritti finora
        long long bytes_written = 0;
        
        /* Si scrive il file in chunk da 1 MB fino a raggiungere str_size_bytes.
         * Si calcola quanti byte mancano da scrivere (remaining) e si determina
         * la dimensione del prossimo chunk (chunk_size). Normalmente è BUFFER_SIZE
         * ma nell'ultimo chunk potrebbe essere minore se i byte rimanenti
         * sono meno di 1 MB. Si riempie il buffer con byte casuali e poi
         * si scrive il buffer sul file.
         */
        while (bytes_written < str_size_bytes) {
            
            /*
             * Calcoliamo quanti byte dobbiamo ancora scrivere.
             */
            long long remaining = str_size_bytes - bytes_written;

            size_t chunk_size = (remaining < BUFFER_SIZE) ? remaining : BUFFER_SIZE;
            
            for (size_t i = 0; i < chunk_size; i++) {
                buffer[i] = random_byte();
            }
            
            size_t written = fwrite(buffer, 1, chunk_size, file);
            
            if (written != chunk_size) {
                fprintf(stderr, "ERRORE: Scrittura incompleta (disco pieno?)\n");
                break;  
            }
            
            // Si aggiorna il contatore dei byte scritti
            bytes_written += written;
            
        }
        
        free(buffer);
        fclose(file);
        
        printf("  Completato: %lld bytes scritti\n\n", bytes_written);
    }
    
    printf("Tutti i file sono stati generati nella directory 'Data'.\n");
    
    return 0;
}