#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ3 Data Preparation Utility
=============================

This script prepares single-cell RNA-seq data for RQ3 cellular resolution analysis.
Converts various formats (Seurat, Scanpy, 10X) to standardized format for the pipeline.

Scientific Rationale:
- Single-cell data comes in diverse formats from different platforms
- Standardization ensures consistent downstream analysis
- Enables integration of data from multiple sources/batches

Author: PhD Dissertation Research
Date: December 2025
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


def prepare_csv_format(
    expression_file: str,
    metadata_file: str,
    output_dir: str,
    gene_column: str = "gene_symbol",
    cell_type_column: str = "cell_type",
    condition_column: str = "diet",
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3
):
    """
    Prepare CSV format data for RQ3 analysis.
    
    Expected Input Format:
    ----------------------
    expression_file: Genes × Cells matrix
      - Rows: genes (with gene_symbol as index or column)
      - Columns: cell identifiers
      - Values: counts or normalized expression
    
    metadata_file: Cell × Metadata table
      - Rows: cell identifiers (matching expression columns)
      - Columns: cell_type, diet, strain, sample_id, etc.
    
    Output Format:
    --------------
    - expression_matrix.csv: Clean gene × cell expression matrix
    - cell_metadata.csv: Validated metadata with required columns
    - qc_metrics.txt: Quality control statistics
    """
    print("[INFO] Preparing CSV format data...")
    
    # Load data
    print(f"[INFO] Loading expression data: {expression_file}")
    expr_df = pd.read_csv(expression_file, index_col=0)
    
    print(f"[INFO] Loading metadata: {metadata_file}")
    meta_df = pd.read_csv(metadata_file, index_col=0)
    
    # Validate
    print("\n[INFO] Validating data...")
    print(f"  Expression shape: {expr_df.shape} (genes × cells)")
    print(f"  Metadata shape: {meta_df.shape} (cells × features)")
    
    # Check overlap
    expr_cells = set(expr_df.columns)
    meta_cells = set(meta_df.index)
    common_cells = expr_cells & meta_cells
    
    print(f"  Common cells: {len(common_cells)}/{len(expr_cells)}")
    
    if len(common_cells) == 0:
        print("[ERROR] No overlapping cells between expression and metadata!")
        sys.exit(1)
    
    # Filter to common cells
    expr_df = expr_df[list(common_cells)]
    meta_df = meta_df.loc[list(common_cells)]
    
    # Check required columns
    required_cols = [cell_type_column, condition_column]
    missing_cols = [col for col in required_cols if col not in meta_df.columns]
    
    if missing_cols:
        print(f"[ERROR] Missing required metadata columns: {missing_cols}")
        print(f"[INFO] Available columns: {list(meta_df.columns)}")
        sys.exit(1)
    
    # QC filtering
    print("\n[INFO] Performing quality control filtering...")
    
    # Filter cells by genes detected
    genes_per_cell = (expr_df > 0).sum(axis=0)
    cells_pass = genes_per_cell >= min_genes_per_cell
    print(f"  Cells passing QC (>={min_genes_per_cell} genes): {cells_pass.sum()}/{len(cells_pass)}")
    
    expr_df = expr_df.loc[:, cells_pass]
    meta_df = meta_df.loc[cells_pass]
    
    # Filter genes by cells detected
    cells_per_gene = (expr_df > 0).sum(axis=1)
    genes_pass = cells_per_gene >= min_cells_per_gene
    print(f"  Genes passing QC (detected in >={min_cells_per_gene} cells): {genes_pass.sum()}/{len(genes_pass)}")
    
    expr_df = expr_df.loc[genes_pass, :]
    
    # Summary statistics
    print("\n[INFO] Final dataset statistics:")
    print(f"  Genes: {expr_df.shape[0]}")
    print(f"  Cells: {expr_df.shape[1]}")
    print(f"  Median genes per cell: {genes_per_cell[cells_pass].median():.0f}")
    print(f"  Median UMI per cell: {expr_df.sum(axis=0).median():.0f}")
    
    print("\n[INFO] Cell type distribution:")
    for ct, count in meta_df[cell_type_column].value_counts().items():
        print(f"  {ct}: {count} cells")
    
    print("\n[INFO] Condition distribution:")
    for cond, count in meta_df[condition_column].value_counts().items():
        print(f"  {cond}: {count} cells")
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    expr_output = os.path.join(output_dir, "expression_matrix.csv")
    meta_output = os.path.join(output_dir, "cell_metadata.csv")
    qc_output = os.path.join(output_dir, "qc_metrics.txt")
    
    print(f"\n[INFO] Saving expression matrix: {expr_output}")
    expr_df.to_csv(expr_output)
    
    print(f"[INFO] Saving metadata: {meta_output}")
    meta_df.to_csv(meta_output)
    
    print(f"[INFO] Saving QC metrics: {qc_output}")
    with open(qc_output, 'w') as f:
        f.write("RQ3 Single-Cell Data Quality Control Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Final dataset:\n")
        f.write(f"  Genes: {expr_df.shape[0]}\n")
        f.write(f"  Cells: {expr_df.shape[1]}\n")
        f.write(f"  Median genes/cell: {genes_per_cell[cells_pass].median():.0f}\n")
        f.write(f"  Median UMI/cell: {expr_df.sum(axis=0).median():.0f}\n")
        f.write(f"  Sparsity: {((expr_df == 0).sum().sum() / expr_df.size * 100):.1f}%\n\n")
        
        f.write("Cell type distribution:\n")
        for ct, count in meta_df[cell_type_column].value_counts().items():
            f.write(f"  {ct}: {count} cells\n")
        
        f.write("\nCondition distribution:\n")
        for cond, count in meta_df[condition_column].value_counts().items():
            f.write(f"  {cond}: {count} cells\n")
    
    print("\n[SUCCESS] Data preparation complete!")
    print(f"[INFO] Use these files for RQ3 analysis:")
    print(f"  python rq3_cellular_resolution_flux_analysis.py \\")
    print(f"    --sc_data {expr_output} \\")
    print(f"    --sc_metadata {meta_output} \\")
    print(f"    --sc_format csv")


def prepare_anndata_format(
    h5ad_file: str,
    output_dir: str,
    cell_type_key: str = "cell_type",
    condition_key: str = "diet",
    layer: Optional[str] = None,
    normalize: bool = False,
    log_transform: bool = False
):
    """
    Prepare AnnData/h5ad format for RQ3 analysis.
    
    This function handles Scanpy-processed data, which is common in
    single-cell workflows.
    """
    print("[INFO] Preparing AnnData format...")
    
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError:
        print("[ERROR] anndata and/or scanpy not installed")
        print("[INFO] Install with: pip install anndata scanpy")
        sys.exit(1)
    
    # Load AnnData
    print(f"[INFO] Loading h5ad file: {h5ad_file}")
    adata = ad.read_h5ad(h5ad_file)
    
    print(f"[INFO] AnnData object:")
    print(f"  Shape: {adata.shape} (cells × genes)")
    print(f"  Variables: {list(adata.var.columns)}")
    print(f"  Observations: {list(adata.obs.columns)}")
    print(f"  Layers: {list(adata.layers.keys()) if adata.layers else 'None'}")
    
    # Check required metadata
    if cell_type_key not in adata.obs.columns:
        print(f"[ERROR] Cell type key '{cell_type_key}' not found in adata.obs")
        print(f"[INFO] Available keys: {list(adata.obs.columns)}")
        sys.exit(1)
    
    if condition_key not in adata.obs.columns:
        print(f"[ERROR] Condition key '{condition_key}' not found in adata.obs")
        print(f"[INFO] Available keys: {list(adata.obs.columns)}")
        sys.exit(1)
    
    # Extract expression matrix
    if layer is not None:
        if layer not in adata.layers:
            print(f"[ERROR] Layer '{layer}' not found")
            print(f"[INFO] Available layers: {list(adata.layers.keys())}")
            sys.exit(1)
        X = adata.layers[layer]
        print(f"[INFO] Using layer: {layer}")
    else:
        X = adata.X
        print(f"[INFO] Using adata.X")
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        print("[INFO] Converting sparse matrix to dense...")
        X = X.toarray()
    
    # Optional normalization
    if normalize:
        print("[INFO] Normalizing to 10,000 UMI per cell...")
        adata_temp = adata.copy()
        if layer is not None:
            adata_temp.X = adata_temp.layers[layer].copy()
        sc.pp.normalize_total(adata_temp, target_sum=1e4)
        if log_transform:
            print("[INFO] Log-transforming (log1p)...")
            sc.pp.log1p(adata_temp)
        X = adata_temp.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
    
    # Create DataFrames
    expr_df = pd.DataFrame(
        X.T,
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    meta_df = adata.obs[[cell_type_key, condition_key]].copy()
    meta_df.columns = ['cell_type', 'diet']  # Standardize names
    
    # Add additional useful metadata if available
    if 'n_genes' in adata.obs.columns:
        meta_df['n_genes_detected'] = adata.obs['n_genes']
    if 'n_counts' in adata.obs.columns:
        meta_df['total_umi'] = adata.obs['n_counts']
    
    # QC summary
    print("\n[INFO] Dataset statistics:")
    print(f"  Genes: {expr_df.shape[0]}")
    print(f"  Cells: {expr_df.shape[1]}")
    print(f"  Cell types: {meta_df['cell_type'].nunique()}")
    print(f"  Conditions: {meta_df['diet'].nunique()}")
    
    print("\n[INFO] Cell type distribution:")
    for ct, count in meta_df['cell_type'].value_counts().items():
        print(f"  {ct}: {count} cells")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    expr_output = os.path.join(output_dir, "expression_matrix.csv")
    meta_output = os.path.join(output_dir, "cell_metadata.csv")
    
    # Also save as h5ad for direct use
    h5ad_output = os.path.join(output_dir, "processed_data.h5ad")
    
    print(f"\n[INFO] Saving expression matrix: {expr_output}")
    expr_df.to_csv(expr_output)
    
    print(f"[INFO] Saving metadata: {meta_output}")
    meta_df.to_csv(meta_output)
    
    print(f"[INFO] Saving processed h5ad: {h5ad_output}")
    adata.write_h5ad(h5ad_output)
    
    print("\n[SUCCESS] Data preparation complete!")


def create_example_data(
    output_dir: str,
    n_cells: int = 1000,
    n_genes: int = 2000,
    cell_types: List[str] = None,
    conditions: List[str] = None
):
    """
    Create example single-cell data for testing RQ3 pipeline.
    
    This generates synthetic data with known properties for validation.
    """
    print("[INFO] Creating example single-cell data...")
    
    if cell_types is None:
        cell_types = ['Hepatocytes', 'Kupffer_cells', 'Stellate_cells', 
                      'Endothelial_cells', 'Cholangiocytes']
    
    if conditions is None:
        conditions = ['SCD', 'HFD', 'WD', 'KD']
    
    np.random.seed(42)
    
    # Generate cell metadata
    n_cells_per_type = n_cells // len(cell_types)
    n_cells_per_condition = n_cells // len(conditions)
    
    cell_ids = [f"Cell_{i:04d}" for i in range(n_cells)]
    
    # Assign cell types (uneven distribution to mimic reality)
    cell_type_assignments = []
    proportions = [0.7, 0.1, 0.1, 0.05, 0.05]  # Hepatocyte-dominant
    for ct, prop in zip(cell_types, proportions):
        n = int(n_cells * prop)
        cell_type_assignments.extend([ct] * n)
    # Fill remaining
    while len(cell_type_assignments) < n_cells:
        cell_type_assignments.append(cell_types[0])
    np.random.shuffle(cell_type_assignments)
    
    # Assign conditions (balanced)
    condition_assignments = []
    for cond in conditions:
        condition_assignments.extend([cond] * n_cells_per_condition)
    while len(condition_assignments) < n_cells:
        condition_assignments.append(conditions[0])
    np.random.shuffle(condition_assignments)
    
    metadata = pd.DataFrame({
        'cell_type': cell_type_assignments,
        'diet': condition_assignments,
        'n_genes': np.random.randint(500, 3000, n_cells),
        'total_umi': np.random.randint(1000, 50000, n_cells)
    }, index=cell_ids)
    
    # Generate expression matrix
    # Different genes have different expression patterns across cell types
    gene_ids = [f"Gene_{i:04d}" for i in range(n_genes)]
    
    expression_data = np.zeros((n_genes, n_cells))
    
    # Cell-type-specific genes (first 500 genes)
    genes_per_type = 100
    for i, ct in enumerate(cell_types):
        ct_cells = [j for j, x in enumerate(cell_type_assignments) if x == ct]
        gene_start = i * genes_per_type
        gene_end = (i + 1) * genes_per_type
        
        # High expression in this cell type
        for g in range(gene_start, min(gene_end, n_genes)):
            expression_data[g, ct_cells] = np.random.negative_binomial(5, 0.3, len(ct_cells))
    
    # Condition-responsive genes (next 400 genes)
    condition_genes_start = 500
    genes_per_condition = 100
    for i, cond in enumerate(conditions):
        cond_cells = [j for j, x in enumerate(condition_assignments) if x == cond]
        gene_start = condition_genes_start + i * genes_per_condition
        gene_end = gene_start + genes_per_condition
        
        for g in range(gene_start, min(gene_end, n_genes)):
            expression_data[g, cond_cells] = np.random.negative_binomial(5, 0.3, len(cond_cells))
    
    # Remaining genes: housekeeping (low variance)
    for g in range(900, n_genes):
        expression_data[g, :] = np.random.negative_binomial(10, 0.5, n_cells)
    
    # Add dropout (sparsity)
    dropout_rate = 0.6
    dropout_mask = np.random.random((n_genes, n_cells)) < dropout_rate
    expression_data[dropout_mask] = 0
    
    expr_df = pd.DataFrame(expression_data, index=gene_ids, columns=cell_ids)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    expr_output = os.path.join(output_dir, "example_expression_matrix.csv")
    meta_output = os.path.join(output_dir, "example_cell_metadata.csv")
    
    print(f"[INFO] Saving example expression matrix: {expr_output}")
    expr_df.to_csv(expr_output)
    
    print(f"[INFO] Saving example metadata: {meta_output}")
    metadata.to_csv(meta_output)
    
    print("\n[SUCCESS] Example data created!")
    print(f"\n[INFO] Test with:")
    print(f"  python rq3_cellular_resolution_flux_analysis.py \\")
    print(f"    --sc_data {expr_output} \\")
    print(f"    --sc_metadata {meta_output} \\")
    print(f"    --model iMM1415.json \\")
    print(f"    --results_dir results_rq3_test")


def main():
    """Command-line interface for data preparation."""
    
    parser = argparse.ArgumentParser(
        description="Prepare single-cell data for RQ3 cellular resolution analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Preparation mode')
    
    # CSV preparation
    csv_parser = subparsers.add_parser('csv', help='Prepare CSV format data')
    csv_parser.add_argument('expression_file', help='Gene × Cell expression matrix CSV')
    csv_parser.add_argument('metadata_file', help='Cell metadata CSV')
    csv_parser.add_argument('--output_dir', default='rq3_prepared_data',
                           help='Output directory')
    csv_parser.add_argument('--cell_type_column', default='cell_type',
                           help='Metadata column for cell types')
    csv_parser.add_argument('--condition_column', default='diet',
                           help='Metadata column for conditions')
    csv_parser.add_argument('--min_genes_per_cell', type=int, default=200,
                           help='Minimum genes per cell for QC')
    csv_parser.add_argument('--min_cells_per_gene', type=int, default=3,
                           help='Minimum cells per gene for QC')
    
    # AnnData preparation
    h5ad_parser = subparsers.add_parser('anndata', help='Prepare AnnData/h5ad format')
    h5ad_parser.add_argument('h5ad_file', help='Path to h5ad file')
    h5ad_parser.add_argument('--output_dir', default='rq3_prepared_data',
                            help='Output directory')
    h5ad_parser.add_argument('--cell_type_key', default='cell_type',
                            help='adata.obs key for cell types')
    h5ad_parser.add_argument('--condition_key', default='diet',
                            help='adata.obs key for conditions')
    h5ad_parser.add_argument('--layer', default=None,
                            help='AnnData layer to use (default: adata.X)')
    h5ad_parser.add_argument('--normalize', action='store_true',
                            help='Normalize to 10,000 UMI per cell')
    h5ad_parser.add_argument('--log_transform', action='store_true',
                            help='Log-transform (log1p) after normalization')
    
    # Example data creation
    example_parser = subparsers.add_parser('example', help='Create example data for testing')
    example_parser.add_argument('--output_dir', default='rq3_example_data',
                               help='Output directory')
    example_parser.add_argument('--n_cells', type=int, default=1000,
                               help='Number of cells')
    example_parser.add_argument('--n_genes', type=int, default=2000,
                               help='Number of genes')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'csv':
            prepare_csv_format(
                expression_file=args.expression_file,
                metadata_file=args.metadata_file,
                output_dir=args.output_dir,
                cell_type_column=args.cell_type_column,
                condition_column=args.condition_column,
                min_genes_per_cell=args.min_genes_per_cell,
                min_cells_per_gene=args.min_cells_per_gene
            )
        
        elif args.command == 'anndata':
            prepare_anndata_format(
                h5ad_file=args.h5ad_file,
                output_dir=args.output_dir,
                cell_type_key=args.cell_type_key,
                condition_key=args.condition_key,
                layer=args.layer,
                normalize=args.normalize,
                log_transform=args.log_transform
            )
        
        elif args.command == 'example':
            create_example_data(
                output_dir=args.output_dir,
                n_cells=args.n_cells,
                n_genes=args.n_genes
            )
    
    except Exception as e:
        print(f"\n[ERROR] Preparation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
