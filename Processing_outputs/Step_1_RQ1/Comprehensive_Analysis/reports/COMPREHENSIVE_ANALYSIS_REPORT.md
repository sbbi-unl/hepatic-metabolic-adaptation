# Comprehensive Metabolic Flux Analysis Report

================================================================================

## Dataset Summary

- **Total Reactions**: 3726
- **Total Samples**: 99
- **Diet Groups**: HFD, KD, SCD, WD
- **Datasets**: GSE101657, GSE159090, GSE160646, GSE188344, GSE246221, GSE248297

## PCA Analysis

- **PC1 Variance**: 9.57%
- **PC2 Variance**: 4.66%
- **Cumulative Variance (PC1+PC2)**: 14.23%

### Centroid Distances (PC1-PC2 space)

| Comparison | Distance |
|------------|----------|
| HFD_vs_SCD | 11.503 |
| KD_vs_SCD | 15.247 |
| WD_vs_SCD | 7.690 |
| HFD_vs_KD | 4.677 |
| HFD_vs_WD | 7.098 |
| KD_vs_WD | 11.772 |

## PERMANOVA Results

### Overall

- **F-statistic**: 2.894
- **p-value**: 0.0010
- **R²**: 0.084

### Pairwise Comparisons

| Contrast | F-statistic | p-value | q-value | R² |
|----------|-------------|---------|---------|----|
| HFD_vs_SCD | 3.912 | 0.0010 | 0.0010 | 0.051 |
| KD_vs_SCD | 3.538 | 0.0010 | 0.0010 | 0.070 |
| WD_vs_SCD | 2.587 | 0.0010 | 0.0010 | 0.052 |
| HFD_vs_KD | 1.874 | 0.0010 | 0.0010 | 0.038 |
| HFD_vs_WD | 2.002 | 0.0010 | 0.0010 | 0.040 |
| KD_vs_WD | 2.759 | 0.0010 | 0.0010 | 0.111 |

## Reaction-Level Differential Analysis

### HFD_vs_SCD

- **Total reactions tested**: 3726
- **Significant reactions**: 104 (2.8%)
- **Upregulated**: 31
- **Downregulated**: 73
- **Mean |Cohen's d|**: 0.098
- **Median |Cohen's d|**: 0.000

### KD_vs_SCD

- **Total reactions tested**: 3726
- **Significant reactions**: 140 (3.8%)
- **Upregulated**: 44
- **Downregulated**: 96
- **Mean |Cohen's d|**: 0.143
- **Median |Cohen's d|**: 0.000

### WD_vs_SCD

- **Total reactions tested**: 3726
- **Significant reactions**: 75 (2.0%)
- **Upregulated**: 30
- **Downregulated**: 45
- **Mean |Cohen's d|**: 0.121
- **Median |Cohen's d|**: 0.000

### HFD_vs_KD

- **Total reactions tested**: 3726
- **Significant reactions**: 24 (0.6%)
- **Upregulated**: 14
- **Downregulated**: 10
- **Mean |Cohen's d|**: 0.115
- **Median |Cohen's d|**: 0.000

### HFD_vs_WD

- **Total reactions tested**: 3726
- **Significant reactions**: 149 (4.0%)
- **Upregulated**: 58
- **Downregulated**: 91
- **Mean |Cohen's d|**: 0.117
- **Median |Cohen's d|**: 0.000

### KD_vs_WD

- **Total reactions tested**: 3726
- **Significant reactions**: 338 (9.1%)
- **Upregulated**: 143
- **Downregulated**: 195
- **Mean |Cohen's d|**: 1.621
- **Median |Cohen's d|**: 0.000

## Subsystem/Pathway Analysis

### HFD_vs_SCD

- **Total subsystems**: 102
- **Sign-consistent subsystems**: 1

**Top 5 Pathways by Effect Size**:

1. Oxidative Phosphorylation: d=-0.559, 3/5 significant
2. Biomass and maintenance functions: d=-0.675, 1/2 significant
3. Tyr, Phe, Trp Biosynthesis: d=0.564, 0/1 significant
4. Cholesterol Metabolism: d=-0.526, 15/43 significant
5. D-alanine metabolism: d=0.126, 0/3 significant

### KD_vs_SCD

- **Total subsystems**: 102
- **Sign-consistent subsystems**: 0

**Top 5 Pathways by Effect Size**:

1. Biomass and maintenance functions: d=-0.821, 1/2 significant
2. Oxidative Phosphorylation: d=-0.791, 1/5 significant
3. Cholesterol Metabolism: d=-0.600, 14/43 significant
4. Salvage Pathway: d=-0.611, 1/3 significant
5. Triacylglycerol Synthesis: d=-0.560, 5/13 significant

### WD_vs_SCD

- **Total subsystems**: 102
- **Sign-consistent subsystems**: 0

**Top 5 Pathways by Effect Size**:

1. Glycolysis/Gluconeogenesis: d=0.142, 8/29 significant
2. Oxidative Phosphorylation: d=-0.496, 1/5 significant
3. Tyr, Phe, Trp Biosynthesis: d=0.463, 0/1 significant
4. R Group Synthesis: d=0.123, 4/50 significant
5. Vitamin A Metabolism: d=0.075, 0/32 significant

### HFD_vs_KD

- **Total subsystems**: 102
- **Sign-consistent subsystems**: 0

**Top 5 Pathways by Effect Size**:

1. Salvage Pathway: d=0.555, 0/3 significant
2. Pyrimidine Catabolism: d=-0.020, 1/18 significant
3. Pentose and Glucuronate Interconversions: d=-0.279, 0/12 significant
4. Cholesterol Metabolism: d=0.162, 0/43 significant
5. Fructose and Mannose Metabolism: d=0.286, 3/16 significant

### HFD_vs_WD

- **Total subsystems**: 102
- **Sign-consistent subsystems**: 0

**Top 5 Pathways by Effect Size**:

1. Glycolysis/Gluconeogenesis: d=-0.132, 7/29 significant
2. Biomass and maintenance functions: d=-0.507, 1/2 significant
3. Glyoxylate and Dicarboxylate Metabolism: d=0.369, 3/14 significant
4. Cholesterol Metabolism: d=-0.366, 15/43 significant
5. Urea cycle/amino group metabolism: d=-0.079, 6/25 significant

### KD_vs_WD

- **Total subsystems**: 102
- **Sign-consistent subsystems**: 1

**Top 5 Pathways by Effect Size**:

1. Glyoxylate and Dicarboxylate Metabolism: d=31.141, 6/14 significant
2. Fructose and Mannose Metabolism: d=-17.047, 4/16 significant
3. Pyruvate Metabolism: d=10.723, 5/27 significant
4. Tyrosine metabolism: d=10.720, 7/45 significant
5. Transport, Extracellular: d=1.728, 81/516 significant

## Pathway Enrichment Analysis

### HFD_vs_SCD

- **Pathways tested**: 25
- **Significantly enriched**: 7

**Top 5 Enriched Pathways**:

1. Oxidative Phosphorylation: 21.50x enrichment (3/5 reactions, q=1.01e-03)
2. Cholesterol Metabolism: 12.50x enrichment (15/43 reactions, q=3.47e-12)
3. Glycolysis/Gluconeogenesis: 8.65x enrichment (7/29 reactions, q=8.46e-05)
4. Triacylglycerol Synthesis: 8.27x enrichment (3/13 reactions, q=1.76e-02)
5. Glycerophospholipid Metabolism: 6.86x enrichment (9/47 reactions, q=5.12e-05)

### KD_vs_SCD

- **Pathways tested**: 32
- **Significantly enriched**: 6

**Top 5 Enriched Pathways**:

1. Triacylglycerol Synthesis: 10.24x enrichment (5/13 reactions, q=6.22e-04)
2. Alanine and Aspartate Metabolism: 8.87x enrichment (4/12 reactions, q=4.78e-03)
3. Cholesterol Metabolism: 8.67x enrichment (14/43 reactions, q=5.76e-09)
4. Glycerophospholipid Metabolism: 5.66x enrichment (10/47 reactions, q=1.02e-04)
5. R Group Synthesis: 4.79x enrichment (9/50 reactions, q=6.22e-04)

### WD_vs_SCD

- **Pathways tested**: 23
- **Significantly enriched**: 3

**Top 5 Enriched Pathways**:

1. Glycolysis/Gluconeogenesis: 13.70x enrichment (8/29 reactions, q=1.30e-06)
2. Glycerophospholipid Metabolism: 8.46x enrichment (8/47 reactions, q=3.55e-05)
3. Transport, Extracellular: 2.12x enrichment (22/516 reactions, q=2.54e-03)

### HFD_vs_KD

- **Pathways tested**: 12
- **Significantly enriched**: 3

**Top 5 Enriched Pathways**:

1. Glyoxylate and Dicarboxylate Metabolism: 33.27x enrichment (3/14 reactions, q=6.62e-04)
2. Fructose and Mannose Metabolism: 29.11x enrichment (3/16 reactions, q=6.62e-04)
3. Tyrosine metabolism: 13.80x enrichment (4/45 reactions, q=6.62e-04)

### HFD_vs_WD

- **Pathways tested**: 35
- **Significantly enriched**: 6

**Top 5 Enriched Pathways**:

1. Cholesterol Metabolism: 8.72x enrichment (15/43 reactions, q=1.07e-09)
2. Starch and Sucrose Metabolism: 7.14x enrichment (6/21 reactions, q=1.42e-03)
3. Glycolysis/Gluconeogenesis: 6.04x enrichment (7/29 reactions, q=1.42e-03)
4. Urea cycle/amino group metabolism: 6.00x enrichment (6/25 reactions, q=3.05e-03)
5. Transport, Peroxisomal: 3.13x enrichment (10/80 reactions, q=6.68e-03)

### KD_vs_WD

- **Pathways tested**: 49
- **Significantly enriched**: 8

**Top 5 Enriched Pathways**:

1. Glyoxylate and Dicarboxylate Metabolism: 4.72x enrichment (6/14 reactions, q=1.39e-02)
2. Alanine and Aspartate Metabolism: 4.59x enrichment (5/12 reactions, q=3.24e-02)
3. Cholesterol Metabolism: 4.36x enrichment (17/43 reactions, q=2.52e-06)
4. Triacylglycerol Synthesis: 4.24x enrichment (5/13 reactions, q=3.40e-02)
5. Glycolysis/Gluconeogenesis: 3.04x enrichment (8/29 reactions, q=3.24e-02)

## Network Analysis

### HFD_vs_SCD

- **Nodes**: 130 (104 reactions, 26 pathways)
- **Edges**: 103
- **Density**: 0.012
- **Connected components**: 27

### KD_vs_SCD

- **Nodes**: 172 (140 reactions, 32 pathways)
- **Edges**: 140
- **Density**: 0.010
- **Connected components**: 32

### WD_vs_SCD

- **Nodes**: 98 (75 reactions, 23 pathways)
- **Edges**: 75
- **Density**: 0.016
- **Connected components**: 23

### HFD_vs_KD

- **Nodes**: 36 (24 reactions, 12 pathways)
- **Edges**: 24
- **Density**: 0.038
- **Connected components**: 12

### HFD_vs_WD

- **Nodes**: 184 (149 reactions, 35 pathways)
- **Edges**: 149
- **Density**: 0.009
- **Connected components**: 35

### KD_vs_WD

- **Nodes**: 387 (338 reactions, 49 pathways)
- **Edges**: 338
- **Density**: 0.005
- **Connected components**: 49

## Generated Files

### CSV Files

- `centroid_distances_PC12.csv`
- `network_metrics.csv`
- `pathway_enrichment_HFD_vs_KD.csv`
- `pathway_enrichment_HFD_vs_SCD.csv`
- `pathway_enrichment_HFD_vs_WD.csv`
- `pathway_enrichment_KD_vs_SCD.csv`
- `pathway_enrichment_KD_vs_WD.csv`
- `pathway_enrichment_WD_vs_SCD.csv`
- `pca_scores.csv`
- `pca_variance_explained.csv`
- `permanova_overall.csv`
- `permanova_pairwise.csv`
- `rank_product_HFD_vs_KD.csv`
- `rank_product_HFD_vs_SCD.csv`
- `rank_product_KD_vs_SCD.csv`
- `reaction_stats_HFD_vs_KD.csv`
- `reaction_stats_HFD_vs_SCD.csv`
- `reaction_stats_HFD_vs_WD.csv`
- `reaction_stats_KD_vs_SCD.csv`
- `reaction_stats_KD_vs_WD.csv`
- `reaction_stats_WD_vs_SCD.csv`
- `sample_metadata.csv`
- `subsystem_analysis_HFD_vs_KD.csv`
- `subsystem_analysis_HFD_vs_SCD.csv`
- `subsystem_analysis_HFD_vs_WD.csv`
- `subsystem_analysis_KD_vs_SCD.csv`
- `subsystem_analysis_KD_vs_WD.csv`
- `subsystem_analysis_WD_vs_SCD.csv`

### Figures

- `Figure_1_PCA_Analysis.png`
- `Figure_2_Volcano_Plots.png`
- `Figure_3_Reaction_Heatmap.png`
- `Figure_4_Subsystem_Enrichment.png`
- `Figure_5_Pathway_Direction.png`
- `Figure_6_Effect_Size_Distributions.png`
- `Figure_7_Network_Visualization.png`

### Network Files (Cytoscape)

- `edges_HFD_vs_KD.csv`
- `edges_HFD_vs_SCD.csv`
- `edges_HFD_vs_WD.csv`
- `edges_KD_vs_SCD.csv`
- `edges_KD_vs_WD.csv`
- `edges_WD_vs_SCD.csv`
- `nodes_HFD_vs_KD.csv`
- `nodes_HFD_vs_SCD.csv`
- `nodes_HFD_vs_WD.csv`
- `nodes_KD_vs_SCD.csv`
- `nodes_KD_vs_WD.csv`
- `nodes_WD_vs_SCD.csv`

================================================================================

**Analysis completed successfully!**
