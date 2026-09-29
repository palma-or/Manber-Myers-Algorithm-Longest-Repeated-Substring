# PALMA ORLANDO 0622702433 p.orlando8@studenti.unisa.it
# Course: High Performance Computing 2025/2026
# Lecturer: Francesco Moscato   fmoscato@unisa.it
#
# Copyright (C) 2025 Palma Orlando
#
# This file is part of ManberMyers_Orlando_Palma_HPC_IZ.
#
# ManberMyers_Orlando_Palma_HPC_IZ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ManberMyers_Orlando_Palma_HPC_IZ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ManberMyers_Orlando_Palma_HPC_IZ.  If not, see <http://www.gnu.org/licenses/>.

import os
import subprocess
import json
import re
import statistics
import sys

# CONFIGURAZIONE GLOBALE
SRC_DIR = "Source"
BIN_DIR = "Build"
DATA_DIR = "Data"
RESULTS_DIR = "Results"
HEADER_DIR = "Headers"

TYPES = ["Sequenziale", "MPI", "CUDA"]
OPTIMIZATIONS = ["-O0", "-O1", "-O2", "-O3"]
MPI_PROCS = [2, 4]
CUDA_BLOCK_SIZES = [32, 64, 128, 256, 512]
NUM_RUNS = 3

SOURCES = {
    "Sequenziale": ["seq_main.c", "seq_suffix_array.c", "common_functions.c"],
    "MPI": ["mpi_main.c", "mpi_suffix_array.c", "merge_algorithms.c", "seq_suffix_array.c"],
    "CUDA": ["cuda_main.cu", "cuda_suffix_array.cu", "seq_suffix_array.c", "common_functions.c"]
}

# GESTIONE DIRECTORY
def ensure_dirs():
    # Crea le directory necessarie per risultati e binari
    if not os.path.exists(RESULTS_DIR): 
        os.makedirs(RESULTS_DIR)
    for t in TYPES:
        path = os.path.join(BIN_DIR, t)
        if not os.path.exists(path): 
            os.makedirs(path)

# PARSING METRICHE DI OUTPUT
def parse_output_metrics(stdout):
    # Estrae le metriche di timing dall'output del programma
    # Supporta formati sia Sequenziale/CUDA che MPI
    metrics = {
        "total": 0.0, "sa_construction": 0.0, 
        "lcp_construction": 0.0, "lrs_search": 0.0, "combined_lcp_lrs": 0.0 
    }
    
    def find_val(pattern, text):
        # Ricerca un valore numerico tramite regex
        match = re.search(pattern, text)
        return float(match.group(1).replace(',', '.')) if match else 0.0

    # Estrae il tempo totale
    metrics["total"] = find_val(r"(?:Totale|TOTALE):\s*(\d+[.,]\d+)", stdout)

    # Differenzia tra formato Sequenziale/CUDA e MPI
    val_sa = find_val(r"Suffix Array(?: \(CUDA\))?:\s*(\d+[.,]\d+)", stdout)
    if val_sa > 0:
        # Formato Sequenziale/CUDA
        metrics["sa_construction"] = val_sa
        metrics["lcp_construction"] = find_val(r"LCP Array(?: \(CUDA/CPU\))?:\s*(\d+[.,]\d+)", stdout)
        metrics["lrs_search"] = find_val(r"LRS Search(?: \(CUDA\))?:\s*(\d+[.,]\d+)", stdout)
    else:
        # Formato MPI
        val_sort = find_val(r"Lettura e Sort:\s*(\d+[.,]\d+)", stdout)
        val_merge = find_val(r"Gather e Merge:\s*(\d+[.,]\d+)", stdout)
        if val_sort > 0:
            metrics["sa_construction"] = val_sort + val_merge
            metrics["combined_lcp_lrs"] = find_val(r"LCP e LRS:\s*(\d+[.,]\d+)", stdout)

    return metrics

# ESECUZIONE COMANDI
def run_command(cmd, shell=True):
    # Esegue un comando shell e restituisce l'output stdout.
    # Ritorna None in caso di errore.
    try:
        result = subprocess.run(cmd, shell=shell, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not result.stdout: 
            return "SUCCESS_NO_OUTPUT"
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRORE COMANDO]: {cmd}")
        if e.stderr: 
            print(f"[STDERR]: {e.stderr.strip()}")
        return None

# COSTRUZIONE PERCORSI FILE
def get_paths(type_name, opt_level, input_file, block_size=None):
    # Genera i percorsi per eseguibile e file di input.
    # Supporta sia Windows che WSL (per MPI).
    ext = ".exe" if type_name != "MPI" else ""
    
    # Il nome dell'eseguibile include il block size per CUDA
    if type_name == "CUDA" and block_size:
        filename = f"{type_name.lower()}_bs{block_size}_{opt_level.replace('-', '')}{ext}"
    else:
        filename = f"{type_name.lower()}_{opt_level.replace('-', '')}{ext}"
    
    win_exe = os.path.join(BIN_DIR, type_name, filename)
    win_input = os.path.join(DATA_DIR, input_file)
    wsl_exe = f"{BIN_DIR}/{type_name}/{filename}"
    wsl_input = f"{DATA_DIR}/{input_file}"
    
    return win_exe, wsl_exe, win_input, wsl_input

# FUNZIONE PRINCIPALE
def main():
    # Orchestrazione principale: compilazione, esecuzione e raccolta risultati.
    ensure_dirs()
    
    # VALIDAZIONE INPUT
    target_file = None
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        print("Uso: python build_manager.py <nome_file.bin>")
        return

    if not os.path.exists(os.path.join(DATA_DIR, target_file)):
        print(f"Errore: File {target_file} non trovato.")
        return

    results_file_path = os.path.join(RESULTS_DIR, f"results_{target_file}.json")
    file_size_mb = os.path.getsize(os.path.join(DATA_DIR, target_file)) / (1024*1024)
    all_results = []

    print(f"\n=== TEST SU: {target_file} ===")

    # FASE 1: COMPILAZIONE
    print("\n1. COMPILAZIONE")
    
    compiled_seq = {}      # [opt] -> True/False
    compiled_mpi = {}      # [opt] -> True/False
    compiled_cuda = {}     # [opt][bs] -> True/False

    for opt in OPTIMIZATIONS:
        print(f"\n> Ottimizzazione {opt}:")
        
        # Compilazione Sequenziale
        win_exe, _, _, _ = get_paths("Sequenziale", opt, "")
        src = " ".join([f"{SRC_DIR}/{f}" for f in SOURCES["Sequenziale"]])
        if run_command(f"gcc -I {HEADER_DIR} {src} {opt} -o {win_exe}"): 
            compiled_seq[opt] = True
            print("  [Seq OK]", end=" ")
        else: 
            compiled_seq[opt] = False
            print("  [Seq FAIL]", end=" ")

        # Compilazione MPI
        _, wsl_exe, _, _ = get_paths("MPI", opt, "")
        src = " ".join([f"{SRC_DIR}/{f}" for f in SOURCES["MPI"]])
        if run_command(f"wsl mpicc -I {HEADER_DIR} {src} {opt} -o {wsl_exe}"):
            compiled_mpi[opt] = True
            print("[MPI OK]")
        else:
            compiled_mpi[opt] = False
            print("[MPI FAIL]")

        # Compilazione CUDA per tutti i block size
        compiled_cuda[opt] = {}
        print("  CUDA:", end=" ")
        for bs in CUDA_BLOCK_SIZES:
            win_exe, _, _, _ = get_paths("CUDA", opt, "", block_size=bs)
            src = " ".join([os.path.join(SRC_DIR, f) for f in SOURCES["CUDA"]])
            if run_command(f"nvcc -I {HEADER_DIR} {src} {opt} -DBLOCK_SIZE={bs} -allow-unsupported-compiler -o {win_exe}"):
                compiled_cuda[opt][bs] = True
                print(f"BS{bs}:OK", end=" ")
            else:
                compiled_cuda[opt][bs] = False
                print(f"BS{bs}:FAIL", end=" ")
        print("")

    # FASE 2: ESECUZIONE E MISURAZIONE
    print("\n2. ESECUZIONE E MISURAZIONE")

    for opt in OPTIMIZATIONS:
        print(f"\n[{opt}]")
        
        # Esecuzione Sequenziale
        if compiled_seq[opt]:
            print(f"   Run Sequenziale...", end=" ", flush=True)
            win_exe, _, win_in, _ = get_paths("Sequenziale", opt, target_file)
            times = []
            for _ in range(NUM_RUNS):
                out = run_command(f"{win_exe} {win_in}")
                if out: 
                    m = parse_output_metrics(out)
                    if m["total"] > 0: 
                        times.append(m)
            
            if times:
                avg = {k: statistics.mean([x[k] for x in times]) for k in times[0]}
                print(f"OK ({avg['total']:.3f}s)")
                all_results.append({
                    "type": "Sequenziale", "opt": opt, "procs": 1, "block_size": None,
                    "metrics_avg": avg
                })
            else: 
                print("FAIL")

        # Esecuzione MPI
        if compiled_mpi[opt]:
            for np in MPI_PROCS:
                print(f"   Run MPI (np={np})...", end=" ", flush=True)
                _, wsl_exe, _, wsl_in = get_paths("MPI", opt, target_file)
                times = []
                for _ in range(NUM_RUNS):
                    out = run_command(f"wsl mpirun -np {np} ./{wsl_exe} {wsl_in}")
                    if out:
                        m = parse_output_metrics(out)
                        if m["total"] > 0: 
                            times.append(m)
                
                if times:
                    avg = {k: statistics.mean([x[k] for x in times]) for k in times[0]}
                    print(f"OK ({avg['total']:.3f}s)")
                    all_results.append({
                        "type": "MPI", "opt": opt, "procs": np, "block_size": None,
                        "metrics_avg": avg
                    })
                else: 
                    print("FAIL")

        # Esecuzione CUDA per tutti i block size compilati
        for bs in CUDA_BLOCK_SIZES:
            if compiled_cuda[opt].get(bs):
                print(f"   Run CUDA (BS={bs})...", end=" ", flush=True)
                win_exe, _, win_in, _ = get_paths("CUDA", opt, target_file, block_size=bs)
                times = []
                for _ in range(NUM_RUNS):
                    out = run_command(f"{win_exe} {win_in}")
                    if out:
                        m = parse_output_metrics(out)
                        if m["total"] > 0: 
                            times.append(m)
                
                if times:
                    avg = {k: statistics.mean([x[k] for x in times]) for k in times[0]}
                    print(f"OK ({avg['total']:.3f}s)")
                    all_results.append({
                        "type": "CUDA", "opt": opt, "procs": 1, "block_size": bs,
                        "metrics_avg": avg
                    })
                else: 
                    print("FAIL")

    # FASE 3: SALVATAGGIO RISULTATI 
    with open(results_file_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n=== RISULTATI SALVATI IN: {results_file_path} ===")

if __name__ == "__main__":
    main()