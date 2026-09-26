# Multi-scale constraint-based modeling of hepatic metabolic adaptation to diet

Code, inputs, and frozen results for

> Madadjim R, Vechetti I, Cui J. *Multi-scale constraint-based modeling reveals conserved and
> context-dependent hepatic metabolic adaptation to diet.* Briefings in Bioinformatics (under revision,
> BIB-26-1735).

The pipeline links four analyses through one shared reaction coordinate in the iMM1415 mouse
genome-scale model:

| Scale | Question | Data | Main output |
|---|---|---|---|
| Bulk tissue | How do HFD, KD and WD reshape hepatic flux? | 6 GEO liver cohorts, 99 samples | 262 diet-responsive reactions; 10-reaction cross-diet signature |
| Genetic background | Which HFD responses are conserved across strains? | GSE182668, 9 founder strains | 243-reaction union; 11-reaction universal core |
| Single cell | Which liver cell types are predicted to contribute? | GSE218300 atlas, 23 cell types | Abundance-weighted attribution (LECs 54.3%) |
| Microbiome | How do gut-derived metabolites modify host flux? | GSE104913 metatranscriptome, AGORA2, MICOM | 47 microbiome-attributable hepatic reactions |

All four scales use one **three-layer E-Flux constraint architecture**:

1. diet-derived exchange bounds;
2. expression-scaled transporter capacities;
3. expression-scaled intracellular enzyme capacities.

Keeping the layers separate means each constraint source can be traced and ablated.

---

## Quick start (about 1 minute, no solver needed)

```bash
git clone https://github.com/sbbi-unl/hepatic-metabolic-adaptation.git
cd hepatic-metabolic-adaptation
pip install pandas numpy scipy matplotlib openpyxl pillow
cp pipeline_config.example.json pipeline_config.json

# Recompute every manuscript headline number from the frozen tables (34 checks)
python run_pipeline.py --config pipeline_config.json --stages verify --use-reference

# Regenerate Figures 2-7 from the frozen tables
python run_pipeline.py --config pipeline_config.json --stages figures --use-reference
```

## Full reproduction (needs Gurobi)

```bash
conda env create -f environment.yml && conda activate hepatic-flux
python run_pipeline.py --config pipeline_config.json --stages preflight
python run_pipeline.py --config pipeline_config.json --stages all                # primary analyses
python run_pipeline.py --config pipeline_config.json --stages all+sensitivity    # + pFBA/FVA sensitivity
python run_pipeline.py --config pipeline_config.json --stages benchmark_ecoli    # optional 13C-MFA benchmark
```

The single-cell matrix (5.7 GB) and the AGORA2 reconstructions are not stored in the git repository.
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) explains where to get them; the Zenodo archive contains the
single-cell matrix.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | Environment, Gurobi licence, solver notes |
| [`docs/PIPELINE_OVERVIEW.md`](docs/PIPELINE_OVERVIEW.md) | Method, stages, parameters, file flow |
| [`docs/REPRODUCE_MANUSCRIPT.md`](docs/REPRODUCE_MANUSCRIPT.md) | Every figure, table and headline number → stage → script → output |
| [`docs/RUN_ON_YOUR_OWN_DATA.md`](docs/RUN_ON_YOUR_OWN_DATA.md) | Input formats and how to apply the pipeline to new data |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | Public datasets, models, external resources, licences |
| [`docs/PROVENANCE_AND_KNOWN_ISSUES.md`](docs/PROVENANCE_AND_KNOWN_ISSUES.md) | Which production run produced each number; known discrepancies |
| [`CHANGELOG.md`](CHANGELOG.md) | Differences from the production working directory |

## Repository layout

```text
run_pipeline.py              one entry point for every stage
pipeline_config.example.json copy to pipeline_config.json and edit paths
configs/                     positive controls, curated species-to-AGORA2 map
inputs/                      models, expression matrices, diet bounds, benchmark inputs
src/
  rq1_bulk/primary_fba/        bulk cohorts, standard FBA (primary)
  rq1_bulk/sensitivity_pfba/   bulk cohorts, pFBA/FVA (sensitivity)
  rq2_strains/primary_fba/     nine strains, standard FBA, tiers and statistics (primary)
  rq2_strains/sensitivity_pfba/
  rq3_single_cell/             pseudobulk pFBA + sensitivity suite; integration/ attribution scripts
  rq4_microbiome/              MICOM community -> hepatic coupling -> attribution
  cross_rq/                    cross-scale tracing (pFBA sensitivity path; legacy 15-reference script)
  benchmarking/                liver and E. coli method benchmarks, ablation, parameter sweep, determinism
  validation/                  positive controls, 13C engine recovery, reviewer closure tests
  reporting/                   figure and supplementary-table scripts
  utils/                       verification, flattening, environment capture
reference_outputs/           frozen tables and figures used in the manuscript + expected numbers
legacy/                      earlier (pFBA-primary) package entry points, kept for provenance
tools/                       Zenodo bundling script
```

## Citation

Please cite the article, and cite this software/data through the Zenodo DOI (see `CITATION.cff`).

## Licence

Code: MIT (`LICENSE`). Data and frozen results produced by this study: CC BY 4.0 (`LICENSE-DATA`).
Third-party inputs keep their original licences (see `docs/DATA_SOURCES.md`).

## Contact

Roland Madadjim (rmadadjim2@huskers.unl.edu), Systems Biology and Biomedical Informatics Laboratory, University of Nebraska–Lincoln.
