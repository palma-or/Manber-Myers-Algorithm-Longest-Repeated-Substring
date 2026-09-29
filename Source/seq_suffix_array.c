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

#include "seq_suffix_array.h"
#include <stdlib.h>
#include <stdio.h>

/* FUNZIONE: compare_suffixes 
 * Questa funzione viene passata a qsort() per ordinare l'array di strutture Suffix.
 * L'ordinamento avviene non confrontando i caratteri, ma confrontando la 
 * coppia di ranghi (rank[0], rank[1]) che riassume l'ordinamento lessicografico 
 * di blocchi di lunghezza crescente.
 * - Criterio Principale (rank[0]): controlla il rank del blocco corrente. 
 *   Se i rank sono diversi, l'ordine è deciso immediatamente.
 * - Criterio Secondario (rank[1]): se i rank correnti sono uguali, si
 *   confronta il rank del blocco successivo per decidere l'ordine. Questo
 *   equivale a confrontare un intero blocco di lunghezza 2h.
 * RISULTATO: restituisce un intero che indica l'ordine tra i due suffissi.
 * - < 0 se il suffisso a viene prima del suffisso b.
 * - = 0 se i suffissi sono considerati uguali nell'ordinamento.
 * - > 0 se il suffisso a viene dopo il suffisso b.
 */
int compare_suffixes(const void *a, const void *b) {
    // const indica che i puntatori non verranno modificati
    const Suffix *s1 = (const Suffix *)a;
    const Suffix *s2 = (const Suffix *)b;
    
    // Calcola la differenza del primo rank
    int diff = s1->rank[0] - s2->rank[0];
    
    // Se i primi rank sono diversi, restituisce la differenza
    // Altrimenti, confronta il secondo rank
    return diff ? diff : (s1->rank[1] - s2->rank[1]);
}

/* ALGORITMO DI MANBER-MYERS
 * La funzione suffixSort implementa l'algoritmo di Manber-Myers, che 
 * costruisce il Suffix Array in tempo O(n log n). L'obiettivo finale è 
 * riempire l'array sa con gli indici di partenza dei suffissi ordinati.
*/
void suffixSort(int *text, int *sa, int n) {
    /* Allocazione strutture di supporto, in partcolare:
     * - suffixes: un array di strutture Suffix che rappresentano i 
     *   suffissi. Viene ordinata ad ogni iterazione per riflettere
     *   l'ordinamento lessicografico basato sui rank correnti.
     * - ind2rank: un array che, dato l'indice di partenza i di un suffisso,
     *   restituisce la sua posizione attuale nell'array suffixes ordinato.
    */
    Suffix *suffixes = (Suffix *)malloc(n * sizeof(Suffix));
    int *ind2rank = (int *)malloc(n * sizeof(int));
    
    if (!suffixes || !ind2rank) {
        fprintf(stderr, "ERRORE: Impossibile allocare memoria in suffixSort\n");
        fprintf(stderr, "Richiesti %zu bytes per suffixes e %zu bytes per ind2rank\n",
                n * sizeof(Suffix), n * sizeof(int));
        return;
    }

    /* 1. Inizializzazione h=1
     * Per ogni suffisso S_i:
     * - suffixes[i].index è posto a i.
     * - suffixes[i].rank[0] è posto al valore intero del carattere text[i].
     * - suffixes[i].rank[1] è posto al valore intero del carattere text[i+1]. 
     * Se i+1 >= n (fine stringa), viene usato un rank speciale -1 come terminatore.
    */
    for (int i = 0; i < n; i++) {
        suffixes[i].index = i;
        suffixes[i].rank[0] = text[i];
        suffixes[i].rank[1] = ((i + 1) < n) ? text[i + 1] : -1;
    }

    /* 2. Viene eseguito quick sort sui suffixes basato sulle coppie di 
     * caratteri. Questo ordina i suffissi in base ai loro primi due caratteri.
     * - suffixes è l'array di strutture Suffix allocato in memoria da ordinare
     * - n è la lunghezza del testo e, di conseguenza, il numero totale di 
     *   suffissi, e quindi il numero di elementi nell'array suffixes.
     * - sizeof(Suffix) è la dimensione in byte di ogni elemento nell'array 
     *   suffixes. qsort utilizza questa informazione per spostarsi nell'array.
     *   Se qsort vuole accedere al (i+1)-esimo elemento, calcola l'indirizzo 
     *   facendo base + (i * size). Questo informa qsort su quanti byte saltare 
     *   per passare da un elemento al successivo durante l'ordinamento.
     * - compare_suffixes è un puntatore alla funzione di confronto che
     *   definisce l'ordine tra due elementi. Questa funzione viene chiamata
     *   ripetutamente da qsort per determinare come ordinare gli elementi.
     */
    qsort(suffixes, n, sizeof(Suffix), compare_suffixes);

    /* 3. Il ciclo parte da h=2 e, ad ogni iterazione, la lunghezza del blocco
    * in esame h, raddoppia (h *= 2). Ad ogni iterazione:
    * - Si assegnano nuovi rank basati sull'ordinamento attuale.
    * - Si aggiorna rank[1] per riflettere i blocchi successivi di lunghezza h.
    * - Si riordina suffixes basato sui nuovi rank.
    * Il ciclo continua finché h non supera 2n. Questo assicura che, alla
    * fine, si stanno confrontando i suffissi su una lunghezza che è almeno
    * n, garantendo l'ordinamento completo. 
    */
    for (int h = 2; h < 2 * n; h *= 2) {
        
        /* 3.1 ASSEGNAZIONE NUOVI RANK
         * Dopo che i suffissi sono stati ordinati per blocchi di lunghezza 
         * h, è necessario assegnare un nuovo rank ai suffissi.
         * Si scorre l'array suffixes già ordinato. Il rank viene incrementato
         * solo se la coppia (rank[0], rank[1]) del suffisso corrente è 
         * diversa dalla coppia del suffisso precedente. Questo assicura che 
         * suffissi lessicograficamente identici (fino alla lunghezza h) abbiano
         * lo stesso rank. Il nuovo rank viene salvato in suffixes[i].rank[0]. 
         * Questo è il rank che riassume il confronto di lunghezza h. Viene 
         * aggiornato l'array inverso ind2rank[suffixes[i].index] = i. 
         * Questo permette di trovare rapidamente il rank di qualsiasi suffisso 
         * dato il suo indice di partenza.
        */ 
        int rank = 0;
        int prev_rank = suffixes[0].rank[0];
        suffixes[0].rank[0] = rank;
        ind2rank[suffixes[0].index] = 0;

        for (int i = 1; i < n; i++) {
            if (suffixes[i].rank[0] == prev_rank && 
                suffixes[i].rank[1] == suffixes[i-1].rank[1]) {
                prev_rank = suffixes[i].rank[0];
                suffixes[i].rank[0] = rank;
            } else {
                prev_rank = suffixes[i].rank[0];
                suffixes[i].rank[0] = ++rank;
            }
            ind2rank[suffixes[i].index] = i;
        }

        // Terminazione anticipata se tutti i rank sono distinti
        if (rank == n - 1) break;

        /* 3.2 AGGIORNAMENTO rank[1] per la prossima iterazione
         * Una vota riordinati i suffissi e assegnati i nuovi rank[0] basati su 
         * blocchi di lunghezza h, è necessario aggiornare rank[1] per 
         * riflettere i blocchi successivi.
         * - next_index calcola l'indice di partenza del blocco successivo a 
         *   distanza h.
         * - ind2rank[next_index] specifica a quale posizione j nell'array 
         *   ordinato si trova il suffisso che inizia a next_index.
         * - suffixes[j].rank[0] è il nuovo rank (calcolato in 3.1) per il 
         *   blocco di lunghezza h che inizia a next_index. Questo rank viene 
         *   copiato in suffixes[i].rank[1].
         * 
         * Quindi, per ogni suffisso si calcola l'indice del suffisso che
         * inizia a i + h. Usando l'array ind2rank, si trova la posizione 
         * di questo suffisso nell'array suffixes ordinato e si assegna
         * il suo rank[0] a suffixes[i].rank[1]. Se next_index supera la 
         * lunghezza del testo n, si assegna -1 come terminatore.
        */
        for (int i = 0; i < n; i++) {
            int next_index = suffixes[i].index + h;
            suffixes[i].rank[1] = (next_index < n) ? 
                                  suffixes[ind2rank[next_index]].rank[0] : -1;
        }

        /* L'array suffixes viene nuovamente ordinato utilizzando la 
         * nuova coppia di rank (rank[0], rank[1]). Poichè rank[0]
         * riassume il confronto sulla lunghessa h e rank[1] riassume 
         * quelllo sulla lunghezza h a partire da i+h, il confronto
         * della coppia ordina i suffissi su una lunghezza totale di 2h.
        */
        qsort(suffixes, n, sizeof(Suffix), compare_suffixes);
    }

    /* 4. COSTRUZIONE Suffix Array FINALE
     * Viene trasferito il risultato dalla struttura suffixes all'array
     * di output sa. Tale array conterrà gli indici di partenza dei
     * suffissi ordinati lessicograficamente.
    */
    for (int i = 0; i < n; i++) {
        sa[i] = suffixes[i].index;
    }

    free(suffixes);
    free(ind2rank);
}

/* ALGORITMO DI KASAI: CALCOLO DELL'LCP ARRAY
 * La funzione getHeight implementa l'algoritmo di Kasai per calcolare
 * l'LCP (Longest Common Prefix) array in tempo O(n). L'LCP array
 * memorizza la lunghezza del prefisso comune più lungo tra suffissi
 * adiacenti nell'ordinamento lessicografico.
 * - text: l'array di interi che rappresenta il testo originale.
 * - sa: il Suffix Array che contiene gli indici di partenza dei suffissi
 *   ordinati lessicograficamente.
 * - lcp: l'array di output che conterrà le lunghezze dei prefissi comuni 
 *   più lunghi tra suffissi adiacenti.
 * - n: la lunghezza del testo, del Suffix Array e dell'LCP array.
 */
void getHeight(int *text, int *sa, int *lcp, int n) {
    /* Nell'algoritmo di Kasai, l'array rank funge da mappa inversa del 
     * Suffix Array. Esso consente una ricerca in tempo O(1) della posizione
     * lessicografica di un suffisso dato il suo indice di partenza.
     * Poichè la lunghezza del testo n viene passata come parametro
     * della funzione, non è possibile dichiarare l'array in modo statico
     * ma si utilizza l'allocazione dinamica.
    */
    int *rank = (int *)malloc(n * sizeof(int));
    if (!rank) {
        fprintf(stderr, "ERRORE: Impossibile allocare memoria per rank in getHeight\n");
        return;
    }

    /* 1. COSTRUZIONE DELL'ARRAY RANK
     * L'array rank viene costruito scorrendo l'Suffix Array sa. Per ogni 
     * suffisso che inizia all'indice sa[i], viene assegnata la posizione i
     * nell'ordinamento lessicografico.
    */
    for (int i = 0; i < n; i++) {
        rank[sa[i]] = i;
    }

    /* h = lunghezza del prefisso comune più lungo trovato nell'iterazione
     * precedente.
    */
    int h = 0;
    lcp[0] = 0;

    /* 2. CALCOLO DELL'LCP ARRAY
     * L'obiettivo è calcolare la luchezza dell'LCP tra ogni suffisso e il 
     * suo predecessore immediato nell'ordinamento lessicografico.
     * Fondamentale è notare che il ciclo non scorre il Suffix Array sa,
     * ma tutti gli indici di partenza dei suffissi nel testo orginale.
    */
    for (int i = 0; i < n; i++) {
        /* 2.1 Verifica del rank
         * rank[i] restituisce la posizione lessicografica del suffisso 
         * che inizia all'indice i. Se rank[i] è 0, significa che il
         * suffisso è il primo nell'ordinamento lessicografico e quindi
         * non ha un predecessore, per cui il suo LCP è definito come 0.
         * In questo caso, si salta il calcolo dell'LCP.   
        */
        if (rank[i] > 0) {
            /* 2.2 Identificazione predecessore nell'ordinamento lessicografico. 
             * L'indice j è l'indice di partenza del suffisso predecessore 
             * che viene definito utilizzando il rank del suffisso i, si 
             * sottrae 1 per ottenere la posizione del predecessore. 
             * Infine, il Suffix Array indica qual è l'indice di partenza 
             * del suffisso rank[i]-1.
            */
            int j = sa[rank[i] - 1];

            /* 2.3 Calcolo LCP tra il suffisso corrente e il suo predecessore
             * La chiave è la variabile h che, all'inizio del ciclo while
             * non è zero, ma contiene la lunghessa del LCP calcolata per un
             * suffisso precedente (suffisso i-1 e il suo predecessore).
             * - i + h < n --> verifica che si è nei limiti del suffisso corrente
             * - j + h < n --> verifica che si è nei limiti del suffisso predecessore
             * - text[i + h] == text[j + h] --> confronta i caratteri alla posizione 
             * corrente i+h e j+h. Se sono uguali, significa che il prefisso comune
             * può essere esteso di un carattere in più, quindi si incrementa h.
             * Il ciclo continua finché i caratteri corrispondenti sono uguali
             * e non si superano i limiti dei suffissi.
             * Quando il loop termina, h contiene la lunghezza dell'LCP. 
            */
            while (i + h < n && j + h < n && text[i + h] == text[j + h]) {
                h++;
            }

            lcp[rank[i]] = h;

            /* 2.4 Preparazione per l'iterazione successiva
             * Il lemma di Kasai dice che l'LCP del suffisso successivo i+1 sarà
             * al massimo l'LCP appena trovato h-1. Quindi, decrementando h, si 
             * imposta il punto di partenza per la successiva iterazione, garantendo 
             * che non si ricomincerà da zero ma da un valore più vicino alla
             * lunghezza effettiva dell'LCP. Ciò è quello che ottimizza la complessità
             * dell'algoritmo da O(n^2) a O(n).
            */
            if (h > 0) h--;
        }
    }
    
    free(rank);
}

/* CALCOLO DELLA LUNGHEZZA DELLA LRS
 * La funzione findLRS calcola la lunghezza della Longest Repeated
 * Substring (LRS) utilizzando l'LCP array. 
 * - lcp: l'LCP array che contiene le lunghezze dei prefissi comuni
 * più lunghi tra suffissi adiacenti nell'ordinamento lessicografico.
 * - n: la lunghezza dell'LCP array.
 * La funzione scorre l'LCP array per trovare il valore massimo,
 * che rappresenta la lunghezza della LRS.  
 * Ha complessità paria a O(n) perchè la funzione esegue un singolo
 * ciclo che itera n-1 volte con operazioni in tempo costante O(1).
 * 
 * Inizia da i = 1 e non da 0 perchè l'elemento lcp[0] è sempre 0 
 * per definizione.
 */
int findLRS(int *lcp, int n) {
    int maxLCP = 0;
    for (int i = 1; i < n; i++) {
        if (lcp[i] > maxLCP) {
            maxLCP = lcp[i];
        }
    }
    return maxLCP;
}