# hepatic-metabolic-adaptation

[![CI](https://github.com/sbbi-unl/hepatic-metabolic-adaptation/actions/workflows/ci.yml/badge.svg)](https://github.com/sbbi-unl/hepatic-metabolic-adaptation/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Reproducibility repository for the manuscript:

> **Multi-scale constraint-based modeling reveals conserved and context-dependent hepatic metabolic adaptation to diet**


This repository contains all analysis code to reproduce the constraint-based metabolic flux modeling results reported across four research questions (RQ1–RQ4), from raw gene expression inputs through publication-ready figures and supplementary tables.

---

## Overview

The pipeline applies **E-Flux2 + parsimonious FBA** to the mouse hepatic genome-scale metabolic model **iMM1415** (3,726 reactions, 1,375 genes) across four layers of biological context:

| Step | Research Question | Biological Layer | Key Input |
|------|---|---|---|
| 1 | RQ1 | Diet effects across 4 diets, 6 pooled GEO cohorts | Bulk RNA-seq (n = 99 samples) |
| 2 | RQ2 | Genetic background across 9 CC/DO founder strains | Bulk RNA-seq, GSE182668 |
| 3 | RQ3 | Cell-type-specific contributions, 23 hepatic populations | scRNA-seq, 32,282 cells |
| 4 | RQ4 | Gut microbiome–hepatic flux interactions | Metatranscriptomics, GSE104913 |

---

## Repository Structure

```
hepatic-metabolic-adaptation/
├── config/
│   ├── environment_gurobi.yml                     # Conda environment specification
│   ├── expanded_diet_bounds_flat.json             # Diet-specific exchange bounds
│   ├── pipeline_config_rq1.yaml                   # RQ1 pipeline parameters
│   ├── rq2_pipeline_config.json                   # RQ2 per-strain runner (9 strains)
│   ├── rq3_pipeline_config.json                   # RQ3 integration pipeline
│   └── rq4_config_updated.json                    # RQ4 microbiome integration
├── data/
│   ├── GSE182668/                                 # Per-strain bulk RNA-seq (RQ2, RQ4)
│   ├── scRNA_52w_data/                            # scRNA-seq expression + metadata (RQ3)
│   ├── GSEMERGED_SCD_HFD_KD_WD_gene_expression.csv  # Pooled multi-cohort RNA-seq (RQ1)
│   ├── iMM1415.json                               # Mouse hepatic GEM (3,726 reactions)
│   ├── Meta_GSE104913.csv                         # Gut metatranscriptomics (RQ4)
│   ├── mouse_entrez_to_symbol.csv                 # Gene ID mapping
│   ├── reaction_annotations.csv                   # iMM1415 reaction metadata
│   └── expanded_diet_bounds_flat.json             # Diet exchange bounds
├── scripts/
│   │
│   │  ── RQ1: Diet-induced flux rewiring ──────────────────────────────────────────
│   ├── run_flux_pipeline_v3.py                    # RQ1 orchestrator (two-branch pipeline)
│   ├── map_fixv5_multigroupsv8_layered_manuscript_run.py  # Core E-Flux2 + pFBA engine
│   ├── batch_correct_flux_by_dataset_3d.py        # Dataset batch-effect removal
│   ├── combined_analysis_interactive_generic.py   # PCA, PERMANOVA, interactive plots
│   ├── comprehensive_flux_analysis.py             # Extended reaction-level statistics
│   │
│   │  ── RQ2: Genetic background ──────────────────────────────────────────────────
│   ├── run_layered_executor.py                    # Config-driven per-strain runner
│   │
│   │  ── RQ3: Cell-type contributions ──────────────────────────────────────────────
│   ├── rq3_data_preparation.py                    # scRNA-seq preprocessing (h5ad → CSV)
│   ├── rq3_FINAL_COMPLETE_REVISED.py              # Single-cell E-Flux2 + flux comparison
│   ├── master_rq_integration_pipeline.py          # RQ3 master orchestrator
│   ├── rq1_rq2_rq3_integration_analysis_REVISED.py
│   ├── rq2_rq3_multi_strain_integration_ENHANCED.py
│   ├── hierarchical_attribution_LOCAL.py          # Function/Location/Lineage attribution
│   ├── hierarchical_figures_LOCAL.py              # Hierarchical attribution figures
│   ├── bulk_validation_LOCAL.py                   # Bulk vs single-cell concordance
│   │
│   │  ── RQ4: Microbiome-hepatic integration ───────────────────────────────────────
│   ├── rq4_master_pipeline_updated.py             # RQ4 orchestrator (stages 1–4)
│   ├── rq4_microbiome_community_modeling_v2026.py # Stage 1: MICOM community flux
│   ├── rq4_hepatic_integration_CORRECTED_v13.py   # Stage 2: Portal metabolites → hepatic
│   ├── rq4_attribution_analysis.py                # Stage 3: Diet vs microbiome decomposition
│   ├── rq4_pathway_enrichment_module.py           # Stage 3b: Pathway enrichment
│   │
│   │  ── Supplementary table generation ─────────────────────────────────────────────
│   ├── generate_RQ1_Convergent_Reactions_Supplementary_Tables.py
│   ├── generate_RQ2_PerStrain_Reactions_Supplementary_Tables.py
│   ├── generate_Section23_RQ3_Supplementary_Tables.py
│   ├── generate_Section24_Supplementary_Tables.py
│   ├── generate_Section25_RQ4_Supplementary_Tables.py
│   └── generate_Section33_Supplementary_Tables.py
│
└── Processing_outputs/                            # Pipeline outputs (not committed)
    ├── Step_1_RQ1/
    ├── Step_2_RQ2/
    ├── Step_3_RQ3/
    └── Step_4_RQ4/
```

---

## Environment Setup

Tested with Python 3.10+ on Linux (Ubuntu 22.04, WSL2) and Windows 11.

```bash
# 1. Clone the repository
git clone https://github.com/sbbi-unl/hepatic-metabolic-adaptation.git
cd hepatic-metabolic-adaptation

# 2. Create and activate the conda environment
conda env create -f config/environment_gurobi.yml
conda activate hepatic-flux
```

### LP Solver

RQ1–RQ3 run with any COBRApy-compatible solver. RQ4 MICOM community modeling performs best with Gurobi.

```bash
# Option A: HiGHS — free, no license required
pip install highspy

# Option B: Gurobi — fastest; free academic license at https://www.gurobi.com/academia/
pip install gurobipy
# Then activate your license: grbgetkey <YOUR_KEY>
```

> **Font/display warnings on Linux**: `findfont: Font family 'Arial' not found` and `qt.qpa.plugin: Could not find the Qt platform plugin "wayland"` are cosmetic and do not affect results. Set `MPLBACKEND=Agg` to suppress the Qt warning on headless servers.

---

## Data

All input files are in `data/`. Large files should be downloaded from the sources below; do not re-upload them to git.

| File | GEO / Source | Size | Used in |
|------|---|---|---|
| `GSEMERGED_SCD_HFD_KD_WD_gene_expression.csv` | GSE101657, GSE159090, GSE160646, GSE188344, GSE246221, GSE248297 | ~26 MB | RQ1 |
| `GSE182668/male-{STRAIN}-GSE182668_HFD_SCD_gene_expression.csv` | [GSE182668](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182668) | — | RQ2, RQ4 |
| `scRNA_52w_data/` | GSE218300 | — | RQ3 |
| `Meta_GSE104913.csv` | [GSE104913](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104913) | ~143 KB | RQ4 |
| `iMM1415.json` | [BiGG / COBRA](http://bigg.ucsd.edu/models/iMM1415) | ~3.6 MB | All |
| `mouse_entrez_to_symbol.csv` | NCBI Gene | ~134 KB | All |
| `reaction_annotations.csv` | Derived from iMM1415 | ~432 KB | RQ4 |
| `expanded_diet_bounds_flat.json` | This repository (`config/`) | ~19 KB | All |
| `data/AGORA2_models/` | [vmh.life](https://www.vmh.life/#microbiome) | ~GB | RQ4 |

> **AGORA2**: Download microbial SBML models from [vmh.life](https://www.vmh.life/#microbiome) and place under `data/AGORA2_models/`. Update `agora_models_dir` in `config/rq4_config_updated.json`.

---

## Running the Pipeline

All steps run from the repository root with the conda environment activated.

---

### Step 1 — RQ1: Diet-Induced Hepatic Flux Rewiring

Runs two parallel branches (uncorrected and batch-corrected E-Flux2/pFBA), then a comprehensive differential flux analysis. Total runtime: **~5–6 minutes** with Gurobi.

```bash
python scripts/run_flux_pipeline_v3.py \
    --results_dir Processing_outputs/Step_1_RQ1 \
    --config config/pipeline_config_rq1.yaml \
    --parallel
```

**Internal stages and timing:**

| Branch | Stage | Script | ~Time |
|--------|-------|--------|-------|
| Uncorrected | E-Flux2 + pFBA | `map_fixv5_multigroupsv8_layered_manuscript_run.py` | 130 s |
| Uncorrected | PCA + PERMANOVA | `combined_analysis_interactive_generic.py` | 6 s |
| Batch-corrected | E-Flux2 + pFBA | `map_fixv5_multigroupsv8_layered_manuscript_run.py` | 131 s |
| Batch-corrected | Batch correction | `batch_correct_flux_by_dataset_3d.py` | 3 s |
| Batch-corrected | PCA + PERMANOVA | `combined_analysis_interactive_generic.py` | 6 s |
| Final | Comprehensive statistics | `comprehensive_flux_analysis.py` | 44 s |

**Key parameters** (from `config/pipeline_config_rq1.yaml`):

```yaml
test_type:               mann-whitney
eflux_quantile:          0.95
eflux_floor:             0.1
eflux_cap:               1000.0
objective_id:            BIOMASS_mm_1_no_glygln
transporter_strategy:    either
edge_abs_diff_threshold: 0.2
rank_product_for:        HFD,KD,WD
```

**Key outputs** in `Processing_outputs/Step_1_RQ1/`:

```
batch_corrected/flux_analysis/
  reaction_flux_comparison_extended_batch_corrected.csv  # All 99 samples x 3,726 reactions
Comprehensive_Analysis/
  RQ1_flux_pairwise_stats.csv         # Pairwise Mann-Whitney U / FDR, all 6 contrasts
  RQ1_permanova_pairwise.csv          # PERMANOVA F, R2, p, q per contrast
  RQ1_rank_product.csv                # Multi-cohort rank-product integration
  RQ1_pathway_enrichment_*.csv        # Per-contrast hypergeometric enrichment
pipeline_manifests/manifest_*.json   # Full reproducibility manifest
```

---

### Step 2 — RQ2: Genetic Background Across 9 Strains

Runs E-Flux2/pFBA for each of 9 CC/DO founder strains (GSE182668, HFD vs SCD).

```bash
python scripts/run_layered_executor.py \
    --config config/rq2_pipeline_config.json
```

Jobs run sequentially by default (`parallel: 1`). To run a subset or resume failed jobs:

```bash
# Specific strains only
python scripts/run_layered_executor.py \
    --config config/rq2_pipeline_config.json \
    --only C57BL6J,AJ

# Resume (skip already-completed strains)
python scripts/run_layered_executor.py \
    --config config/rq2_pipeline_config.json \
    --resume
```

**Expected outputs** per strain in `Processing_outputs/Step_2_RQ2/{STRAIN}/`:

```
rank_product.csv
flux_analysis/reaction_flux_comparison_extended.csv
stats_comparison/flux_pairwise_stats.csv
cytoscape_edges/*.csv
```

Run summary log: `Processing_outputs/Step_2_RQ2/logs_layered_runs/run_summary_*.csv`

---

### Step 3 — RQ3: Cell-Type-Specific Contributions

Applies E-Flux2/pFBA to 23 hepatic cell populations from a 52-week Western Diet scRNA-seq dataset (32,282 cells × 14,161 genes; 1,075 genes matched to iMM1415; 1,962 reactions constrained per solve).

**3a. Prepare expression matrices** (skip if `data/scRNA_52w_data/` is already populated):

```bash
python scripts/rq3_data_preparation.py anndata data/scRNA_52w_WD.h5ad \
    --output_dir data/scRNA_52w_data \
    --cell_type_key cell_annotation \
    --condition_key disease \
    --normalize \
    --log_transform
```

**3b. Run master integration pipeline:**

```bash
python scripts/master_rq_integration_pipeline.py \
    --run_all \
    --config config/rq3_pipeline_config.json
```

This runs three sub-steps automatically (Generate → Interpret → Generalize), then calls `hierarchical_attribution_LOCAL.py`, `hierarchical_figures_LOCAL.py`, and `bulk_validation_LOCAL.py`.

**Hierarchical attribution results (from actual run):**

| Hierarchy | Dominant Group | Contribution |
|-----------|---------------|-------------|
| Function | Structural cells (LECs, HSCs) | 67.5% |
| Location | Sinusoidal zone | 74.8% |
| Lineage | Endothelial | 53.8% |

**Key outputs** in `Processing_outputs/Step_3_RQ3/`:

```
statistics/statistical_tests.csv                   # 85,698 reaction x cell-type pairs
Step_3b_results_rq1_rq3_integration/tables/
  phase2_contribution_analysis.csv                 # Abundance-weighted contribution scores
  cell_abundance_used.csv                          # 23 cell types with proportions
Hierarchical_Analysis/
  function_attribution.csv
  location_attribution.csv
  lineage_attribution.csv
  hierarchical_results.json
  hierarchical_pie_charts.png                      # 300 DPI main-text figure
  bulk_singlecell_validation.png                   # Pearson r = -0.190, p < 0.001
```

---

### Step 4 — RQ4: Gut Microbiome–Hepatic Integration

Integrates MICOM community modeling (AGORA2) with iMM1415 to decompose hepatic flux into diet (~82.6%) and microbiome (~17.4%) components across 415 significant reactions.

```bash
python scripts/rq4_master_pipeline_updated.py \
    --config config/rq4_config_updated.json
```

**Pipeline stages:**

| Stage | Script | Description |
|-------|--------|-------------|
| 1 — Community Modeling | `rq4_microbiome_community_modeling_v2026.py` | MICOM cooperative tradeoff = 0.5; ND-SCD vs DD-HFD |
| 2 — Hepatic Integration | `rq4_hepatic_integration_CORRECTED_v13.py` | Portal metabolites as exchange constraints (scaling = 0.1) |
| 3 — Attribution | `rq4_attribution_analysis.py` | Factorial variance decomposition: diet vs microbiome |
| 3b — Pathway Enrichment | `rq4_pathway_enrichment_module.py` | Hypergeometric enrichment, 103 subsystems, 7 compartments |
| 4 — Reporting | *(internal)* | Integrated analysis report |

> **Solver**: Stage 1 requires Gurobi or CPLEX. Set `"solver": "glpk"` in `config/rq4_config_updated.json` for a free fallback (slower, may fail for large communities).

**Key outputs** in `Processing_outputs/Step_4_RQ4/`:

```
01_community_modeling/portal_metabolites_for_hepatic_model.json
02_hepatic_integration/
  condition_ND_SCD/ND_SCD_flux_comparison.csv
  condition_DD_HFD/DD_HFD_flux_comparison.csv
03_attribution_analysis/
  flux_attribution_analysis.csv          # 3,726 reactions, diet + microbiome delta-flux
  pathway_enrichment/
    pathway_enrichment_results.csv
    compartment_enrichment_results.csv
    pathway_synergy_analysis.csv         # 319 synergistic, 96 antagonistic
    PATHWAY_ANALYSIS_SUMMARY.txt
05_analysis_reports/RQ4_Analysis_Report.txt
```

---

### Supplementary Table Generation

After all four steps complete, the supplementary Excel tables are regenerated from pipeline outputs:

```bash
python scripts/generate_RQ1_Convergent_Reactions_Supplementary_Tables.py
python scripts/generate_RQ2_PerStrain_Reactions_Supplementary_Tables.py
python scripts/generate_Section23_RQ3_Supplementary_Tables.py
python scripts/generate_Section24_Supplementary_Tables.py
python scripts/generate_Section25_RQ4_Supplementary_Tables.py
python scripts/generate_Section33_Supplementary_Tables.py
```

---

## Key Results Summary

| RQ | Finding | Value |
|----|---------|-------|
| RQ1 | Diet-responsive reactions (FDR < 0.05, multi-cohort rank-product) | 262 |
| RQ1 | Overall PERMANOVA (diet effect) | p = 0.001 |
| RQ1 | Strongest pairwise separation | KD vs WD (R2 = 0.140) |
| RQ2 | Strain-responsive reactions (HFD vs SCD) | 243 |
| RQ2 | Conserved core reactions (9/9 strains) | 11 |
| RQ3 | Hepatic cell populations modeled | 23 |
| RQ3 | Dominant functional contributor | Structural cells (67.5%) |
| RQ3 | Dominant spatial contributor | Sinusoidal zone (74.8%) |
| RQ3 | Dominant lineage contributor | Endothelial (53.8%) |
| RQ4 | Reactions with significant microbiome effect | 415 |
| RQ4 | Variance explained by diet | ~82.6% |
| RQ4 | Variance explained by microbiome | ~17.4% |

---

## Troubleshooting

**`findfont: Font family 'Arial' not found`** — Cosmetic warning. Figures render correctly. Fix: `sudo apt-get install ttf-mscorefonts-installer && fc-cache -f`.

**`qt.qpa.plugin: Could not find the Qt platform plugin "wayland"`** — Expected on headless servers. Set `export MPLBACKEND=Agg` before running.

**Gurobi license error** — Free academic licenses at [gurobi.com/academia](https://www.gurobi.com/academia/). Use `"solver": "glpk"` as a free fallback.

**MICOM convergence warnings** — Default tradeoff is 0.5. If convergence fails, try 0.3 or 0.7 in `config/rq4_config_updated.json`.

**Memory during RQ3** — Processing 23 cell types requires ~8–12 GB RAM.

---

## Citation

```bibtex
@article{[citation_key],
  title   = {Diet- and Genotype-Dependent Metabolic Flux Rewiring Revealed
             by Multi-Cohort Constraint-Based Modeling},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Correspondence

Please open a [GitHub Issue](https://github.com/sbbi-unl/hepatic-metabolic-adaptation/issues) for questions about the code.

Systems Biology and Biomedical Informatics (SBBI) Lab
University of Nebraska-Lincoln
