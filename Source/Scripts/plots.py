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

"""
SCRIPT PER GENERAZIONE GRAFICI ANALISI PRESTAZIONI
GRAFICI GENERATI:
-----------------
1. Speedup_All_Opts_Bars.png              - Speedup per tutte le ottimizzazioni (MPI vs CUDA)
2. Efficiency_All_Opts.png                - Efficienza parallela MPI (2 e 4 procs)
3. Optimization_Comparison_Bars_100MB.png - Confronto ottimizzazioni (barre raggruppate, 100 MB)
4. Absolute_Time_Grouped.png              - Tempo assoluto per dimensione (4 subplot)
5. MPI_vs_CUDA_Direct_Comparison.png      - Confronto diretto MPI vs CUDA (-O3)
6. Compiler_Impact_Comparison.png         - Impatto ottimizzazioni compilatore
7. CUDA_BlockSize_Heatmap.png             - Heatmap tuning block size CUDA
8. CUDA_BlockSize_Speedup.png             - Speedup relativo tra block sizes
9. Phase_Breakdown_Comparison.png         - Breakdown temporale per fase

ORGANIZZAZIONE:
---------------
- Sezione 1: Caricamento Dati e Utility
- Sezione 2: Grafici Comparativi Multi-Ottimizzazione
- Sezione 3: Grafici Confronto Diretto
- Sezione 4: Grafici Analisi CUDA
- Sezione 5: Grafici Breakdown
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
import glob
import numpy as np

# CONFIGURAZIONE GLOBALE
SEARCH_DIRS = [".", "Results"]
OUTPUT_DIR = os.path.join("Results", "plots")

sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)

# Colori per configurazioni
CONFIG_COLORS = {
    "Sequenziale": "#34495e",
    "MPI (2 procs)": "#e67e22",
    "MPI (4 procs)": "#d35400",
    "CUDA": "#27ae60"
}

def get_config_color(label):
    """Restituisce il colore corretto gestendo le varianti CUDA"""
    if label.startswith("CUDA"):
        return CONFIG_COLORS["CUDA"]
    return CONFIG_COLORS.get(label, "#333333")

# Colori per ottimizzazioni
OPT_COLORS = {
    "-O0": "#e74c3c",
    "-O1": "#f39c12",
    "-O2": "#3498db",
    "-O3": "#2ecc71"
}

# Colori per block size CUDA (gradiente verde)
BS_COLORS = {
    32: "#d5f4e6",
    64: "#a9dfbf",
    128: "#7dcea0",
    256: "#52be80",
    512: "#27ae60"
}


# SEZIONE 1: UTILITY E CARICAMENTO DATI
def ensure_dir(directory):
    """Crea directory se non esiste"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_size_from_filename(filename):
    """Estrae dimensione file dal nome"""
    match = re.search(r"string_(\d+)MB", filename)
    return int(match.group(1)) if match else None

def get_label(row):
    """Genera label descrittiva per configurazione"""
    if row['type'] == 'Sequenziale': 
        return "Sequenziale"
    if row['type'] == 'MPI': 
        return f"MPI ({int(row['procs'])} procs)"
    if row['type'] == 'CUDA': 
        return f"CUDA (BS={int(row['block_size'])})"
    return "Unknown"

def load_data():
    """Carica tutti i file JSON dei risultati"""
    all_data = []
    found_files = []
    
    for d in SEARCH_DIRS:
        if os.path.exists(d):
            found_files.extend(glob.glob(os.path.join(d, "results_string_*MB*.json")))
    
    found_files = list(set(found_files))
    print(f"Trovati {len(found_files)} file di risultati.")
    
    for filepath in found_files:
        filename = os.path.basename(filepath)
        size = extract_size_from_filename(filename)
        if size is None: 
            continue
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            for entry in data:
                entry['size_mb'] = size
                entry['label'] = get_label(entry)
                entry['total_time'] = entry['metrics_avg']['total']
                all_data.append(entry)
        except Exception as e: 
            print(f"Errore caricamento {filename}: {e}")
            
    return pd.DataFrame(all_data)

def filter_best_cuda(df):
    """Mantiene solo il miglior block size CUDA per ogni (size, opt)"""
    rows = []
    for (size, opt), group in df.groupby(['size_mb', 'opt']):
        # Mantiene tutte le righe NON-CUDA (Sequenziale, MPI)
        rows.append(group[group['type'] != 'CUDA'])
        
        # Gestione CUDA: trova il migliore
        cuda_group = group[group['type'] == 'CUDA']
        if not cuda_group.empty:
            best_idx = cuda_group['total_time'].idxmin()
            best_row = cuda_group.loc[[best_idx]].copy()
            rows.append(best_row)
            
    return pd.concat(rows, ignore_index=True)

def get_cuda_labels(df):
    """Trova tutte le label CUDA presenti nel dataframe"""
    return [l for l in df['label'].unique() if l.startswith('CUDA')]


# SEZIONE 2: GRAFICI COMPARATIVI MULTI-OTTIMIZZAZIONE
def plot_speedup_all_opts_bars(df):
    """
    GRAFICO 1: Speedup per TUTTE le ottimizzazioni - Barre Raggruppate
    Mostra MPI (4 procs) e CUDA side-by-side con 4 barre per ottimizzazione
    """
    subset_clean = filter_best_cuda(df)
    
    # Identifica le configurazioni da plottare
    # MPI fisso, per CUDA prendiamo tutte le varianti trovate (che saranno le "best")
    mpi_config = 'MPI (4 procs)'
    
    # Creiamo una colonna 'Group' per semplificare il plotting side-by-side
    # Tutti i CUDA finiranno nel gruppo "CUDA" ma manterremo il dettaglio se serve
    subset_clean['plot_group'] = subset_clean['type'].apply(
        lambda x: 'CUDA' if x == 'CUDA' else x
    )
    # Aggiorniamo per MPI specifici
    subset_clean.loc[subset_clean['label'] == mpi_config, 'plot_group'] = mpi_config
    
    groups_to_plot = [mpi_config, 'CUDA']
    
    speedup_data = []
    for size, group_size in subset_clean.groupby('size_mb'):
        for opt, group_opt in group_size.groupby('opt'):
            seq = group_opt[group_opt['type'] == 'Sequenziale']
            if seq.empty: continue
            t_seq = seq.iloc[0]['total_time']
            
            for _, row in group_opt.iterrows():
                if row['plot_group'] in groups_to_plot:
                    speedup_data.append({
                        'size_mb': size,
                        'opt': opt,
                        'label': row['label'], 
                        'plot_group': row['plot_group'],
                        'speedup': t_seq / row['total_time']
                    })
    
    df_spd = pd.DataFrame(speedup_data)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, group in enumerate(groups_to_plot):
        ax = axes[idx]
        data_group = df_spd[df_spd['plot_group'] == group]
        
        sizes = sorted(data_group['size_mb'].unique())
        x = np.arange(len(sizes))
        width = 0.2
        
        opts = ['-O0', '-O1', '-O2', '-O3']
        for i, opt in enumerate(opts):
            data_opt = data_group[data_group['opt'] == opt].sort_values('size_mb')
            if not data_opt.empty:
                offset = width * (i - 1.5)
                bars = ax.bar(x + offset, data_opt['speedup'], width,
                             label=opt, color=OPT_COLORS[opt],
                             edgecolor='black', linewidth=1)
                
                # Annotazioni barre
                for bar, val, label in zip(bars, data_opt['speedup'], data_opt['label']):
                    if val > 0.3:
                        text = f'{val:.2f}'
                        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                               text, ha='center', va='bottom',
                               fontsize=7, fontweight='bold')
        
        ax.axhline(1, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Dimensione File (MB)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Speedup (×)', fontweight='bold', fontsize=12)
        
        # Titolo differenziato
        title = group
        if group == 'CUDA':
            title = "CUDA (Best Config)"
            
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.grid(axis='y', alpha=0.4)
        ax.legend(title='Ottimizzazione', fontsize=9)
    
    fig.suptitle('Speedup vs Sequenziale - Tutte le Ottimizzazioni',
                 fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Speedup_All_Opts_Bars.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

def plot_efficiency_all_opts(df):
    """
    GRAFICO 2: Efficienza MPI per tutte le ottimizzazioni
    Due subplot: MPI (2 procs) e MPI (4 procs)
    """
    subset_clean = filter_best_cuda(df)
    
    efficiency_data = []
    for size, group_size in subset_clean.groupby('size_mb'):
        for opt, group_opt in group_size.groupby('opt'):
            seq = group_opt[group_opt['type'] == 'Sequenziale']
            if seq.empty: continue
            t_seq = seq.iloc[0]['total_time']
            
            for _, row in group_opt.iterrows():
                if row['type'] == 'MPI':
                    speedup = t_seq / row['total_time']
                    efficiency = (speedup / row['procs']) * 100
                    efficiency_data.append({
                        'size_mb': size,
                        'opt': opt,
                        'procs': row['procs'],
                        'efficiency': efficiency
                    })
    
    df_eff = pd.DataFrame(efficiency_data)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, procs in enumerate([2, 4]):
        ax = axes[idx]
        data_procs = df_eff[df_eff['procs'] == procs]
        
        sizes = sorted(data_procs['size_mb'].unique())
        x = np.arange(len(sizes))
        width = 0.2
        
        opts = ['-O0', '-O1', '-O2', '-O3']
        for i, opt in enumerate(opts):
            data_opt = data_procs[data_procs['opt'] == opt].sort_values('size_mb')
            if not data_opt.empty:
                offset = width * (i - 1.5)
                bars = ax.bar(x + offset, data_opt['efficiency'], width,
                             label=opt, color=OPT_COLORS[opt],
                             edgecolor='black', linewidth=1)
                
                for bar, val in zip(bars, data_opt['efficiency']):
                    if val > 10:
                        ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                               f'{val:.0f}%', ha='center', va='bottom',
                               fontsize=7, fontweight='bold')
        
        ax.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Dimensione File (MB)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Efficienza (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'MPI ({procs} procs)', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.4)
        ax.legend(title='Ottimizzazione', fontsize=9)
    
    fig.suptitle('Efficienza Parallela MPI - Tutte le Ottimizzazioni',
                 fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Efficiency_All_Opts.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimization_comparison_bars(df):
    """
    GRAFICO 3: Confronto Ottimizzazioni - Barre Raggruppate (100 MB)
    Più intuitivo della heatmap: mostra speedup per ogni config x opt
    """
    subset_100 = df[df['size_mb'] == 100].copy()
    subset_clean = filter_best_cuda(subset_100)
    
    # Identifica dinamicamente le configurazioni presenti
    # Vogliamo: MPI (2), MPI (4), e la CUDA (Best)
    available_configs = []
    
    # Cerca MPI
    for p in [2, 4]:
        lbl = f"MPI ({p} procs)"
        if not subset_clean[subset_clean['label'] == lbl].empty:
            available_configs.append(lbl)
            
    # Cerca CUDA
    cuda_labels = get_cuda_labels(subset_clean)
    available_configs.extend(cuda_labels) 
    
    opts = ['-O0', '-O1', '-O2', '-O3']
    
    speedup_data = []
    for opt in opts:
        group_opt = subset_clean[subset_clean['opt'] == opt]
        seq = group_opt[group_opt['type'] == 'Sequenziale']
        if seq.empty: continue
        t_seq = seq.iloc[0]['total_time']
        
        for config in available_configs:
            data_config = group_opt[group_opt['label'] == config]
            if not data_config.empty:
                speedup = t_seq / data_config.iloc[0]['total_time']
                speedup_data.append({
                    'opt': opt,
                    'config': config,
                    'speedup': speedup
                })
    
    if not speedup_data:
        print("Dati insufficienti per grafico ottimizzazioni")
        return
    
    df_spd = pd.DataFrame(speedup_data)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Lista unica di configurazioni sull'asse X
    unique_configs = sorted(list(set(df_spd['config'])))
    # Ordiniamo per mettere MPI prima di CUDA
    unique_configs.sort(key=lambda x: 1 if "CUDA" in x else 0)
    
    x = np.arange(len(unique_configs))
    width = 0.2
    
    for i, opt in enumerate(opts):
        data_opt = df_spd[df_spd['opt'] == opt]
        
        # Allineamento dati
        speedups = []
        for c in unique_configs:
            row = data_opt[data_opt['config'] == c]
            if not row.empty:
                speedups.append(row.iloc[0]['speedup'])
            else:
                speedups.append(0)
        
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, speedups, width,
                     label=opt, color=OPT_COLORS[opt],
                     edgecolor='black', linewidth=1)
        
        # Annotazioni
        for bar, val in zip(bars, speedups):
            if val > 0.2:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                       f'{val:.2f}×', ha='center', va='bottom',
                       fontsize=8, fontweight='bold')
    
    ax.axhline(1, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label='Baseline (Sequenziale)')
    
    ax.set_xlabel('Configurazione', fontweight='bold', fontsize=13)
    ax.set_ylabel('Speedup vs Sequenziale (×)', fontweight='bold', fontsize=13)
    ax.set_title('Confronto Ottimizzazioni per Configurazione (100 MB)',
                fontweight='bold', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_configs, rotation=15, ha='right')
    ax.legend(title='Ottimizzazione', fontsize=10, ncol=2, loc='upper left')
    ax.grid(axis='y', alpha=0.4)
    ax.set_ylim(0, max(df_spd['speedup']) * 1.15)
    
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Optimization_Comparison_Bars_100MB.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

def plot_absolute_time_grouped(df):
    """
    GRAFICO 4: Tempo Assoluto per Dimensione - 4 Subplot
    Mostra tempo assoluto per 50, 100, 200, 500 MB
    """
    subset_clean = filter_best_cuda(df)
    
    # Definisci le configurazioni base
    base_configs = ['Sequenziale', 'MPI (4 procs)']
    
    sizes_to_plot = [50, 100, 200, 500]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes_to_plot):
        ax = axes[idx]
        data_size = subset_clean[subset_clean['size_mb'] == size]
        
        # Costruiamo lista dinamica labels
        current_configs = []
        for c in base_configs:
            if not data_size[data_size['label'] == c].empty:
                current_configs.append(c)
        
        has_cuda = not data_size[data_size['type'] == 'CUDA'].empty
        if has_cuda:
            current_configs.append('CUDA')
            
        opts = ['-O0', '-O1', '-O2', '-O3']
        x = np.arange(len(current_configs))
        width = 0.2
        
        for i, opt in enumerate(opts):
            data_opt = data_size[data_size['opt'] == opt]
            times = []
            labels_detail = []
            
            for config in current_configs:
                if config == 'CUDA':
                    row = data_opt[data_opt['type'] == 'CUDA']
                    if not row.empty:
                        times.append(row.iloc[0]['total_time'])
                        labels_detail.append(row.iloc[0]['label']) 
                    else:
                        times.append(0)
                        labels_detail.append("")
                else:
                    row = data_opt[data_opt['label'] == config]
                    if not row.empty:
                        times.append(row.iloc[0]['total_time'])
                        labels_detail.append(config)
                    else:
                        times.append(0)
                        labels_detail.append("")
            
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, times, width, label=opt,
                         color=OPT_COLORS[opt], edgecolor='black', linewidth=1)
            
            for bar, val, lbl in zip(bars, times, labels_detail):
                if val > 0.5:
                    txt = f'{val:.1f}s'
                    if "CUDA" in lbl:
                        match = re.search(r"BS=(\d+)", lbl)
                        if match:
                            txt += f"\n(BS={match.group(1)})"
                    
                    ax.text(bar.get_x() + bar.get_width()/2, val + max(times)*0.02,
                           txt, ha='center', va='bottom',
                           fontsize=6, fontweight='bold', rotation=0)
        
        ax.set_xlabel('Configurazione', fontweight='bold', fontsize=11)
        ax.set_ylabel('Tempo Esecuzione (s)', fontweight='bold', fontsize=11)
        ax.set_title(f'{size} MB', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(current_configs, rotation=20, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.4)
        if idx == 0:
            ax.legend(title='Opt', loc='upper left', fontsize=8)
    
    fig.suptitle('Tempo Assoluto per Dimensione e Ottimizzazione',
                 fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Absolute_Time_Grouped.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


# SEZIONE 3: GRAFICI CONFRONTO DIRETTO
def plot_mpi_vs_cuda_direct(df):
    """
    GRAFICO 5: Confronto Diretto MPI vs CUDA (-O3)
    Barre side-by-side per visualizzare gap prestazionale
    """
    subset = df[df['opt'] == '-O3'].copy()
    subset_clean = filter_best_cuda(subset)
    
    # Identifica le config: MPI 4 procs e le CUDA trovate
    configs_to_plot = ['MPI (4 procs)']
    cuda_labels = get_cuda_labels(subset_clean)
    configs_to_plot.extend(cuda_labels)
    
    # Filtriamo
    subset_plot = subset_clean[subset_clean['label'].isin(configs_to_plot)]
    
    speedup_data = []
    for size, group in subset_plot.groupby('size_mb'):
        seq = subset_clean[(subset_clean['size_mb'] == size) & 
                          (subset_clean['type'] == 'Sequenziale')]
        if seq.empty: continue
        t_seq = seq.iloc[0]['total_time']
        
        for _, row in group.iterrows():
            speedup_data.append({
                'size_mb': size,
                'label': row['label'],
                'speedup': t_seq / row['total_time']
            })
    
    df_spd = pd.DataFrame(speedup_data)
    
    if df_spd.empty:
        print("  Nessun dato disponibile per il confronto MPI vs CUDA (-O3)")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    
    sizes = sorted(df_spd['size_mb'].unique())
    x = np.arange(len(sizes))
    width = 0.35
        
    # Serie MPI
    mpi_label = 'MPI (4 procs)'
    data_mpi = df_spd[df_spd['label'] == mpi_label]
    spd_map_mpi = dict(zip(data_mpi['size_mb'], data_mpi['speedup']))
    aligned_mpi = [spd_map_mpi.get(s, 0) for s in sizes]
    
    ax.bar(x - width/2, aligned_mpi, width,
           label=mpi_label, color=CONFIG_COLORS[mpi_label],
           edgecolor='black', linewidth=1.5)
    
    # Annotazioni MPI
    for i, val in enumerate(aligned_mpi):
        if val > 0:
            ax.text(x[i] - width/2, val + 0.05, f'{val:.2f}×',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Serie CUDA (aggregata)
    aligned_cuda = []
    cuda_labels_for_legend = set()
    bar_labels = [] 
    
    for s in sizes:
        # Trova la riga CUDA per questa size
        row = df_spd[(df_spd['size_mb'] == s) & (df_spd['label'].str.startswith('CUDA'))]
        if not row.empty:
            aligned_cuda.append(row.iloc[0]['speedup'])
            lbl = row.iloc[0]['label']
            bar_labels.append(lbl)
            cuda_labels_for_legend.add("CUDA (Best Config)")
        else:
            aligned_cuda.append(0)
            bar_labels.append("")

    bars_cuda = ax.bar(x + width/2, aligned_cuda, width,
           label="CUDA", color=CONFIG_COLORS['CUDA'],
           edgecolor='black', linewidth=1.5)
    
    # Annotazioni CUDA con BS
    for i, (bar, val, lbl) in enumerate(zip(bars_cuda, aligned_cuda, bar_labels)):
        if val > 0:
            bs_text = ""
            match = re.search(r"BS=(\d+)", lbl)
            if match:
                bs_text = f"\n(BS={match.group(1)})"
            
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                   f'{val:.2f}×{bs_text}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')
    
    ax.axhline(1, color='gray', linestyle='--', linewidth=2, alpha=0.7,
              label='Baseline Sequenziale')
    
    ax.set_xlabel('Dimensione File (MB)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Speedup vs Sequenziale (×)', fontweight='bold', fontsize=13)
    ax.set_title('Confronto Diretto: MPI vs CUDA (Ottimizzazione -O3)',
                fontweight='bold', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.4)
    
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "MPI_vs_CUDA_Direct_Comparison.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

def plot_compiler_impact(df):
    """
    GRAFICO 6: Impatto Ottimizzazioni Compilatore
    Confronto -O0 vs -O1 vs -O2 vs -O3 su configurazione sequenziale (100 MB)
    """
    subset = df[(df['size_mb'] == 100) & (df['type'] == 'Sequenziale')].copy()
    
    if subset.empty: return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Tempo assoluto
    opts = ['-O0', '-O1', '-O2', '-O3']
    times = [subset[subset['opt'] == opt].iloc[0]['total_time'] for opt in opts]
    
    bars = ax1.bar(opts, times, color=[OPT_COLORS[opt] for opt in opts],
                   edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.2f}s', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Livello Ottimizzazione', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Tempo Esecuzione (s)', fontweight='bold', fontsize=12)
    ax1.set_title('Tempo Assoluto', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.4)
    
    # Subplot 2: Miglioramento relativo vs -O0
    t_o0 = times[0]
    improvements = [(t_o0 - t) / t_o0 * 100 for t in times]
    
    bars = ax2.bar(opts, improvements, color=[OPT_COLORS[opt] for opt in opts],
                   edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Livello Ottimizzazione', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Miglioramento vs -O0 (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Miglioramento Relativo', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.4)
    ax2.set_ylim(0, max(improvements) + 5)
    
    fig.suptitle('Impatto Ottimizzazione Compilatore (Sequenziale, 100 MB)',
                 fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Compiler_Impact_Comparison.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


# SEZIONE 4: GRAFICI ANALISI CUDA
def plot_cuda_blocksize_heatmap(df):
    """
    GRAFICO 7: Heatmap Block Size CUDA (-O3)
    Righe = dimensioni, Colonne = block sizes
    """
    cuda_only = df[(df['type'] == 'CUDA') & (df['opt'] == '-O3')].copy()
    if cuda_only.empty:
        print("Nessun dato CUDA per heatmap")
        return
    
    pivot = cuda_only.pivot_table(
        index='size_mb',
        columns='block_size',
        values='total_time',
        aggfunc='mean'
    )
    
    if pivot.empty: return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f'{int(bs)}' for bs in pivot.columns], fontsize=11)
    ax.set_yticklabels([f'{int(s)} MB' for s in pivot.index], fontsize=11)
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > pivot.values.max()*0.6 else 'black'
                ax.text(j, i, f'{val:.2f}s',
                       ha='center', va='center', color=color,
                       fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Block Size', fontweight='bold', fontsize=13)
    ax.set_ylabel('Dimensione Input', fontweight='bold', fontsize=13)
    ax.set_title('Heatmap Tempo Esecuzione CUDA - Tuning Block Size (-O3)',
                fontweight='bold', fontsize=15, pad=15)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Tempo (s)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "CUDA_BlockSize_Heatmap.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cuda_blocksize_speedup(df):
    """
    GRAFICO 8: Speedup Relativo tra Block Sizes
    Reference: BS=256 (baseline = 1.0x)
    """
    cuda_only = df[(df['type'] == 'CUDA') & (df['opt'] == '-O3')].copy()
    if cuda_only.empty: return
    
    speedup_data = []
    for size in sorted(cuda_only['size_mb'].unique()):
        data_size = cuda_only[cuda_only['size_mb'] == size]
        baseline = data_size[data_size['block_size'] == 256]
        if baseline.empty: continue
        t_baseline = baseline.iloc[0]['total_time']
        
        for _, row in data_size.iterrows():
            bs = int(row['block_size'])
            speedup_data.append({
                'size_mb': size,
                'block_size': bs,
                'speedup': t_baseline / row['total_time']
            })
    
    if not speedup_data: return
    df_spd = pd.DataFrame(speedup_data)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sizes = sorted(df_spd['size_mb'].unique())
    x = np.arange(len(sizes))
    width = 0.15
    
    block_sizes = sorted(df_spd['block_size'].unique())
    for i, bs in enumerate(block_sizes):
        data_bs = df_spd[df_spd['block_size'] == bs].sort_values('size_mb')
        if not data_bs.empty:
            offset = width * (i - len(block_sizes)/2 + 0.5)
            bars = ax.bar(x + offset, data_bs['speedup'], width,
                         label=f'BS={bs}', color=BS_COLORS.get(bs, '#27ae60'),
                         edgecolor='black', linewidth=1)
            
            for bar, val in zip(bars, data_bs['speedup']):
                if abs(val - 1.0) > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                           f'{val:.2f}×', ha='center', va='bottom',
                           fontsize=7, fontweight='bold')
    
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label='Baseline (BS=256)')
    
    ax.set_xlabel('Dimensione File (MB)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Speedup Relativo vs BS=256', fontweight='bold', fontsize=13)
    ax.set_title('Confronto Block Size CUDA - Speedup Relativo (-O3)',
                fontweight='bold', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left', fontsize=10, ncol=3)
    ax.grid(axis='y', alpha=0.4)
    
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "CUDA_BlockSize_Speedup.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


# SEZIONE 5: GRAFICI BREAKDOWN
def plot_phase_breakdown(df):
    """
    GRAFICO 9: Breakdown Temporale per Fase (100 MB, -O3)
    Stacked bar chart mostrando SA Construction, LCP, LRS, Other
    """
    subset = df[(df['size_mb'] == 100) & (df['opt'] == '-O3')].copy()
    subset_clean = filter_best_cuda(subset)
    
    # Identifica configurazioni da mostrare: Sequenziale, MPI 2/4, CUDA (con BS)
    base_configs = ['Sequenziale', 'MPI (2 procs)', 'MPI (4 procs)']
    configs_to_plot = []
    
    for c in base_configs:
        if not subset_clean[subset_clean['label'] == c].empty:
            configs_to_plot.append(c)
            
    cuda_labels = get_cuda_labels(subset_clean)
    configs_to_plot.extend(cuda_labels)
    
    subset_plot = subset_clean[subset_clean['label'].isin(configs_to_plot)]
    
    if subset_plot.empty: return
    
    # Estrai metriche per fase
    phase_data = []
    for _, row in subset_plot.iterrows():
        metrics = row['metrics_avg']
        total = metrics['total']
        
        sa_time = metrics.get('sa_construction', 0)
        lcp_time = metrics.get('combined_lcp_lrs', 0)
        other_time = total - sa_time - lcp_time
        
        phase_data.append({
            'label': row['label'],
            'SA Construction': (sa_time / total) * 100,
            'LCP/LRS': (lcp_time / total) * 100,
            'Other/Merge': (other_time / total) * 100
        })
    
    df_phases = pd.DataFrame(phase_data)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    phases = ['SA Construction', 'LCP/LRS', 'Other/Merge']
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    
    # Ordiniamo per consistenza
    configs_ordered = [c for c in configs_to_plot if c in df_phases['label'].values]
    
    x = np.arange(len(configs_ordered))
    bottom = np.zeros(len(configs_ordered))
    
    for phase, color in zip(phases, colors):
        values = [df_phases[df_phases['label'] == c][phase].values[0] 
                 if not df_phases[df_phases['label'] == c].empty else 0 
                 for c in configs_ordered]
        
        bars = ax.bar(x, values, 0.6, label=phase, bottom=bottom,
                     color=color, edgecolor='white', linewidth=2)
        
        # Annotazioni percentuali
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 5:  # Solo se >5% per leggibilità
                ax.text(bar.get_x() + bar.get_width()/2, bottom[i] + val/2,
                       f'{val:.0f}%', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
        
        bottom += values
    
    ax.set_xlabel('Configurazione', fontweight='bold', fontsize=13)
    ax.set_ylabel('Percentuale Tempo (%)', fontweight='bold', fontsize=13)
    ax.set_title('Breakdown Temporale per Fase (100 MB, -O3)',
                fontweight='bold', fontsize=15, pad=15)
    ax.set_xticks(x)
    # Mostra label incluse BS per CUDA
    ax.set_xticklabels(configs_ordered, rotation=20, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.4)
    
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Phase_Breakdown_Comparison.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


# MAIN
def main():
    """Funzione principale - genera tutti i grafici essenziali"""

    print(" GENERAZIONE GRAFICI ANALISI PRESTAZIONI MANBER-MYERS")
    ensure_dir(OUTPUT_DIR)
    
    df = load_data()
    if df.empty:
        print("Nessun dato trovato. Verifica che i file JSON siano presenti.")
        return
    print(f"Caricati {len(df)} record da {df['size_mb'].nunique()} dimensioni")
    
    plot_speedup_all_opts_bars(df)
    plot_efficiency_all_opts(df)
    plot_optimization_comparison_bars(df)
    plot_absolute_time_grouped(df)
    
    plot_mpi_vs_cuda_direct(df)
    plot_compiler_impact(df)
    
    plot_cuda_blocksize_heatmap(df)
    plot_cuda_blocksize_speedup(df)
    plot_phase_breakdown(df)

    print(f"Tutti i grafici salvati in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()