#!/usr/bin/env python3
"""
Usage:
    python comprehensive_flux_analysis.py --input <flux_file.csv> --output <output_dir>
"""

import argparse
import os
import sys
import re
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, hypergeom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import networkx as nx
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

# Configuration
RANDOM_SEED = 42
N_PERMUTATIONS = 999
FDR_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.5
DPI = 300
FIGURE_FORMAT = ['png', 'pdf']

# Set random seeds
np.random.seed(RANDOM_SEED)

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================================================
# COMPARISON ORDERING HELPER
# ==============================================================================

def generate_pairwise_comparisons(groups, control_group='SCD'):
    """
    Generate pairwise comparisons with control group always as reference (second position).
    
    This ensures consistent treatment-vs-control labeling where SCD (or specified control)
    always appears as the second element, making it clear that other groups are treatments
    being compared against the baseline.
    
    Parameters:
    -----------
    groups : list
        List of group names (e.g., ['HFD', 'KD', 'SCD', 'WD'])
    control_group : str, default='SCD'
        Name of the control/baseline group
    
    Returns:
    --------
    list of tuples
        List of (treatment, control) pairs followed by treatment-treatment pairs
    
    Example:
    --------
    >>> groups = ['HFD', 'KD', 'SCD', 'WD']
    >>> generate_pairwise_comparisons(groups, control_group='SCD')
    [('HFD', 'SCD'), ('KD', 'SCD'), ('WD', 'SCD'), ('HFD', 'KD'), ('HFD', 'WD'), ('KD', 'WD')]
    
    Notes:
    ------
    - First generates all treatment vs control comparisons
    - Then generates all treatment vs treatment comparisons
    - Ensures control never appears as first element (except in treatment-treatment pairs)
    - Maintains consistent ordering for reproducibility
    """
    groups = sorted(groups)  # Ensure consistent ordering
    comparisons = []
    
    # First: All treatments vs control
    if control_group in groups:
        for g in groups:
            if g != control_group:
                comparisons.append((g, control_group))
    
    # Second: All non-control pairwise comparisons
    non_control_groups = [g for g in groups if g != control_group]
    for g1, g2 in combinations(non_control_groups, 2):
        comparisons.append((g1, g2))
    
    return comparisons

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def setup_output_dir(output_dir):
    """Create output directory structure"""
    output_dir = Path(output_dir)
    subdirs = {
        'csv': output_dir / 'csv_outputs',
        'figures': output_dir / 'figures',
        'networks': output_dir / 'cytoscape_networks',
        'reports': output_dir / 'reports'
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, subdirs

def standardize_group_name(name):
    """Standardize diet group names"""
    name = str(name).upper().strip()
    
    # Define canonical names and aliases
    aliases = {
        'SCD': ['SCD', 'STANDARD', 'STD', 'CHOW', 'CONTROL', 'CTL'],
        'HFD': ['HFD', 'HIGH_FAT', 'HIGHFAT', 'HF'],
        'KD': ['KD', 'KETO', 'KETOGENIC'],
        'WD': ['WD', 'WESTERN', 'WES']
    }
    
    for canonical, group_aliases in aliases.items():
        pattern = r"(?:^|_)(" + "|".join(re.escape(a) for a in group_aliases) + r")(?![A-Z0-9])"
        if re.search(pattern, name, flags=re.I):
            return canonical
    
    # Fallback: extract first part before underscore
    return name.split("_", 1)[0]

def parse_dataset_id(col_name):
    """Extract dataset ID from column name"""
    match = re.search(r"(?:GSE|GSM)\d+", str(col_name), flags=re.I)
    return match.group(0).upper() if match else "UNKNOWN"

def z_score_normalize(X):
    """Z-score normalize columns"""
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return X / sd

def cohen_d(x, y):
    """Calculate Cohen's d effect size"""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(x) - np.mean(y)) / pooled_std

def save_figure(fig, filename, subdirs, dpi=DPI):
    """Save figure in multiple formats"""
    for fmt in FIGURE_FORMAT:
        filepath = subdirs['figures'] / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_prepare_data(input_file):
    """Load flux data and prepare sample metadata"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} reactions")
    print(f"Total columns: {len(df.columns)}")
    
    # Identify sample columns (exclude ReactionID, ReactionName, Subsystem, Mean/Std columns, and derived DIFF/RATIO columns)
    exclude_patterns = ['ReactionID', 'ReactionName', 'Subsystem', 'MeanFlux', 'StdFlux', '_N$', '_Mean$', '_Std$', '^DIFF\\(', '^RATIO\\(']
    exclude_regex = '|'.join(exclude_patterns)
    
    sample_cols = [col for col in df.columns 
                   if not re.search(exclude_regex, col, re.I)]
    
    print(f"Identified {len(sample_cols)} sample columns")
    
    # Build metadata
    metadata = []
    for col in sample_cols:
        group = standardize_group_name(col)
        dataset = parse_dataset_id(col)
        metadata.append({
            'SampleID': col,
            'Group': group,
            'Dataset': dataset
        })
    
    metadata_df = pd.DataFrame(metadata)
    
    print("\nSample distribution:")
    print(metadata_df.groupby('Group').size())
    print("\nDataset distribution:")
    print(metadata_df.groupby('Dataset').size())
    print("\nGroup × Dataset:")
    print(pd.crosstab(metadata_df['Dataset'], metadata_df['Group']))
    
    # Extract annotation and flux data
    annotation_cols = ['ReactionID', 'ReactionName', 'Subsystem']
    annotations = df[annotation_cols].copy()
    
    flux_data = df[sample_cols].values.T  # Transpose: samples × reactions
    
    return df, annotations, flux_data, metadata_df, sample_cols

# ==============================================================================
# PCA ANALYSIS
# ==============================================================================

def perform_pca(flux_data, metadata_df):
    """Perform PCA analysis"""
    print("\n" + "="*80)
    print("PCA ANALYSIS")
    print("="*80)
    
    # Z-score normalization
    flux_zscore = z_score_normalize(flux_data)
    
    # PCA
    pca = PCA(n_components=min(20, flux_data.shape[0], flux_data.shape[1]))
    pca_scores = pca.fit_transform(flux_zscore)
    
    # Variance explained
    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)
    
    print(f"PC1 variance: {var_explained[0]:.1%}")
    print(f"PC2 variance: {var_explained[1]:.1%}")
    print(f"Cumulative variance (PC1+PC2): {cum_var[1]:.1%}")
    
    # Calculate centroid distances
    groups = metadata_df['Group'].unique()
    centroids = {}
    
    for group in groups:
        group_mask = metadata_df['Group'] == group
        group_scores = pca_scores[group_mask, :2]
        centroids[group] = np.mean(group_scores, axis=0)
    
    # Pairwise centroid distances
    centroid_distances = {}
    # Use generate_pairwise_comparisons to ensure SCD is control (FIXED)
    for g1, g2 in generate_pairwise_comparisons(groups, control_group='SCD'):
        dist = euclidean(centroids[g1], centroids[g2])
        centroid_distances[f"{g1}_vs_{g2}"] = dist
        #print(f"Distance {g1} ↔ {g2}: {dist:.3f}")
        print(f"Distance {g1} -- {g2}: {dist:.3f}")
    
    results = {
        'pca': pca,
        'scores': pca_scores,
        'var_explained': var_explained,
        'cum_var': cum_var,
        'centroids': centroids,
        'centroid_distances': centroid_distances,
        'flux_zscore': flux_zscore
    }
    
    return results

def calculate_confidence_ellipse(x, y, ax, n_std=1.96, **kwargs):
    """Calculate and plot confidence ellipse"""
    if len(x) < 2:
        return
    
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    mean_x, mean_y = np.mean(x), np.mean(y)
    
    ellipse = Ellipse((mean_x, mean_y),
                      width=ell_radius_x * 2 * scale_x,
                      height=ell_radius_y * 2 * scale_y,
                      **kwargs)
    
    ax.add_patch(ellipse)

# ==============================================================================
# PERMANOVA
# ==============================================================================

def calculate_distance_matrix(Y):
    """Calculate Euclidean distance matrix"""
    diffs = Y[:, None, :] - Y[None, :, :]
    return np.sqrt((diffs * diffs).sum(axis=2))

def permanova(Y, labels, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """Perform PERMANOVA test"""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    D = calculate_distance_matrix(Y)
    N = D.shape[0]
    groups = np.unique(labels)
    k = len(groups)
    
    if k < 2 or N < 3:
        return {'F': np.nan, 'p': np.nan, 'R2': np.nan, 'k': int(k), 'N': int(N)}
    
    # Calculate sum of squares
    A = D ** 2
    SST = A.sum() / N
    
    SSW = 0.0
    for g in groups:
        idx = np.where(labels == g)[0]
        ng = len(idx)
        if ng > 1:
            Ag = A[np.ix_(idx, idx)]
            SSW += Ag.sum() / ng
    
    SSB = max(0.0, SST - SSW)
    
    # F-statistic
    df_between = k - 1
    df_within = N - k
    
    if SSW > 0 and df_within > 0:
        F_obs = (SSB / df_between) / (SSW / df_within)
    else:
        F_obs = np.nan
    
    # R-squared
    R2 = SSB / SST if SST > 0 else 0
    
    # Permutation test
    count = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        SSW_perm = 0.0
        for g in groups:
            idx = np.where(perm_labels == g)[0]
            ng = len(idx)
            if ng > 1:
                Ag = A[np.ix_(idx, idx)]
                SSW_perm += Ag.sum() / ng
        
        SSB_perm = max(0.0, SST - SSW_perm)
        
        if SSW_perm > 0 and df_within > 0:
            F_perm = (SSB_perm / df_between) / (SSW_perm / df_within)
        else:
            F_perm = -np.inf
        
        if not np.isnan(F_obs) and F_perm >= F_obs:
            count += 1
    
    p_value = (count + 1) / (n_perm + 1)
    
    return {
        'F': float(F_obs),
        'p': float(p_value),
        'R2': float(R2),
        'k': int(k),
        'N': int(N)
    }

def perform_permanova_analysis(pca_results, metadata_df):
    """Perform overall and pairwise PERMANOVA"""
    print("\n" + "="*80)
    print("PERMANOVA ANALYSIS")
    print("="*80)
    
    flux_zscore = pca_results['flux_zscore']
    groups = metadata_df['Group'].values
    
    # Overall PERMANOVA
    print("\nOverall PERMANOVA:")
    overall_result = permanova(flux_zscore, groups)
    print(f"  F = {overall_result['F']:.3f}")
    print(f"  p = {overall_result['p']:.4f}")
    print(f"  R² = {overall_result['R2']:.3f}")
    
    # Pairwise PERMANOVA
    print("\nPairwise PERMANOVA:")
    unique_groups = sorted(metadata_df['Group'].unique())
    pairwise_results = []
    
    # Use generate_pairwise_comparisons to ensure SCD is control (FIXED)
    comparisons_list = generate_pairwise_comparisons(unique_groups, control_group='SCD')
    
    for g1, g2 in comparisons_list:
        mask = metadata_df['Group'].isin([g1, g2])
        Y_sub = flux_zscore[mask]
        labels_sub = metadata_df.loc[mask, 'Group'].values
        
        result = permanova(Y_sub, labels_sub)
        result['Contrast'] = f"{g1}_vs_{g2}"
        result['Group1'] = g1
        result['Group2'] = g2
        
        pairwise_results.append(result)
        print(f"  {g1} vs {g2}: F={result['F']:.3f}, p={result['p']:.4f}, R²={result['R2']:.3f}")
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    # FDR correction
    if len(pairwise_df) > 0:
        _, q_values, _, _ = multipletests(pairwise_df['p'], method='fdr_bh')
        pairwise_df['q_value'] = q_values
    
    return overall_result, pairwise_df

# ==============================================================================
# REACTION-LEVEL STATISTICS
# ==============================================================================

def benjamini_hochberg_fdr(p_values):
    """Benjamini-Hochberg FDR correction"""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    
    q_values = np.empty(n)
    prev = 1.0
    
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        q_values[i] = val
        prev = val
    
    out = np.empty(n)
    out[order] = q_values
    
    return out

def calculate_reaction_statistics(df, metadata_df, sample_cols, annotations):
    """Calculate per-reaction statistics for all pairwise contrasts"""
    print("\n" + "="*80)
    print("REACTION-LEVEL STATISTICS")
    print("="*80)
    
    unique_groups = sorted(metadata_df['Group'].unique())
    # Use generate_pairwise_comparisons to ensure SCD is control (FIXED)
    contrasts = generate_pairwise_comparisons(unique_groups, control_group='SCD')
    
    all_stats = {}
    
    for g1, g2 in contrasts:
        contrast_name = f"{g1}_vs_{g2}"
        print(f"\nAnalyzing {contrast_name}...")
        
        # Get sample indices
        mask1 = metadata_df['Group'] == g1
        mask2 = metadata_df['Group'] == g2
        
        cols1 = metadata_df.loc[mask1, 'SampleID'].tolist()
        cols2 = metadata_df.loc[mask2, 'SampleID'].tolist()
        
        results = []
        
        for idx, row in df.iterrows():
            rxn_id = row['ReactionID']
            rxn_name = row['ReactionName']
            subsystem = row['Subsystem']
            
            # Convert to numeric and remove NaN values
            vals1 = pd.to_numeric(row[cols1], errors='coerce').values
            vals2 = pd.to_numeric(row[cols2], errors='coerce').values
            
            vals1 = vals1[~np.isnan(vals1)]
            vals2 = vals2[~np.isnan(vals2)]
            
            if len(vals1) < 2 or len(vals2) < 2:
                continue
            
            # Welch's t-test
            t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
            
            # Cohen's d
            effect_size = cohen_d(vals1, vals2)
            
            # Mean difference
            mean_diff = np.mean(vals1) - np.mean(vals2)
            
            results.append({
                'ReactionID': rxn_id,
                'ReactionName': rxn_name,
                'Subsystem': subsystem,
                f'{g1}_mean': np.mean(vals1),
                f'{g2}_mean': np.mean(vals2),
                'MeanDiff': mean_diff,
                'log2FC': np.log2((np.mean(vals1) + 1e-10) / (np.mean(vals2) + 1e-10)),
                't_statistic': t_stat,
                'p_value': p_val,
                'Cohen_d': effect_size,
                f'{g1}_n': len(vals1),
                f'{g2}_n': len(vals2)
            })
        
        stats_df = pd.DataFrame(results)
        
        if len(stats_df) > 0:
            # FDR correction
            stats_df['q_value'] = benjamini_hochberg_fdr(stats_df['p_value'])
            stats_df['Significant'] = (stats_df['q_value'] < FDR_THRESHOLD) & (np.abs(stats_df['Cohen_d']) > EFFECT_SIZE_THRESHOLD)
            
            # Direction
            stats_df['Direction'] = stats_df['MeanDiff'].apply(lambda x: 'Up' if x > 0 else 'Down')
            
            # Sort by absolute effect size
            stats_df = stats_df.sort_values('Cohen_d', key=abs, ascending=False)
            
            n_sig = stats_df['Significant'].sum()
            n_up = ((stats_df['Significant']) & (stats_df['MeanDiff'] > 0)).sum()
            n_down = ((stats_df['Significant']) & (stats_df['MeanDiff'] < 0)).sum()
            
            print(f"  Total reactions tested: {len(stats_df)}")
            print(f"  Significant (q<{FDR_THRESHOLD}, |d|>{EFFECT_SIZE_THRESHOLD}): {n_sig}")
            print(f"  Upregulated in {g1}: {n_up}")
            print(f"  Downregulated in {g1}: {n_down}")
            print(f"  Mean |Cohen's d|: {stats_df['Cohen_d'].abs().mean():.3f}")
            print(f"  Median |Cohen's d|: {stats_df['Cohen_d'].abs().median():.3f}")
        
        all_stats[contrast_name] = stats_df
    
    return all_stats

# ==============================================================================
# SUBSYSTEM ANALYSIS
# ==============================================================================

def analyze_subsystems(reaction_stats):
    """Analyze subsystem-level patterns"""
    print("\n" + "="*80)
    print("SUBSYSTEM ANALYSIS")
    print("="*80)
    
    subsystem_results = {}
    
    for contrast, stats_df in reaction_stats.items():
        print(f"\nAnalyzing {contrast}...")
        
        if len(stats_df) == 0:
            continue
        
        subsystem_data = []
        
        for subsystem in stats_df['Subsystem'].unique():
            subsys_df = stats_df[stats_df['Subsystem'] == subsystem]
            
            n_reactions = len(subsys_df)
            n_sig = subsys_df['Significant'].sum()
            
            if n_reactions == 0:
                continue
            
            # Sign consistency
            n_positive = (subsys_df['MeanDiff'] > 0).sum()
            n_negative = (subsys_df['MeanDiff'] < 0).sum()
            
            # Binomial test for sign consistency
            if n_positive + n_negative > 0:
                p_binom = stats.binomtest(max(n_positive, n_negative), 
                                         n_positive + n_negative, 
                                         p=0.5, 
                                         alternative='greater').pvalue
            else:
                p_binom = 1.0
            
            sign_consistency = max(n_positive, n_negative) / (n_positive + n_negative) if (n_positive + n_negative) > 0 else 0
            
            # Summary statistics
            mean_effect = subsys_df['Cohen_d'].mean()
            median_effect = subsys_df['Cohen_d'].median()
            mean_abs_effect = subsys_df['Cohen_d'].abs().mean()
            
            subsystem_data.append({
                'Subsystem': subsystem,
                'N_reactions': n_reactions,
                'N_significant': n_sig,
                'Frac_significant': n_sig / n_reactions if n_reactions > 0 else 0,
                'N_positive': n_positive,
                'N_negative': n_negative,
                'Sign_consistency': sign_consistency,
                'p_sign': p_binom,
                'Mean_Cohen_d': mean_effect,
                'Median_Cohen_d': median_effect,
                'Mean_abs_Cohen_d': mean_abs_effect
            })
        
        subsystem_df = pd.DataFrame(subsystem_data)
        
        if len(subsystem_df) > 0:
            # FDR correction for sign test
            subsystem_df['q_sign'] = benjamini_hochberg_fdr(subsystem_df['p_sign'])
            subsystem_df['Sign_consistent'] = subsystem_df['q_sign'] < FDR_THRESHOLD
            
            subsystem_df = subsystem_df.sort_values('Mean_abs_Cohen_d', ascending=False)
            
            print(f"  Total subsystems: {len(subsystem_df)}")
            print(f"  Sign-consistent subsystems (q<{FDR_THRESHOLD}): {subsystem_df['Sign_consistent'].sum()}")
            print(f"  Top 5 subsystems by effect:")
            for _, row in subsystem_df.head(5).iterrows():
                print(f"    {row['Subsystem']}: d={row['Mean_Cohen_d']:.3f}, "
                      f"sign={row['Sign_consistency']:.2f}, {row['N_significant']}/{row['N_reactions']} sig")
        
        subsystem_results[contrast] = subsystem_df
    
    return subsystem_results

# ==============================================================================
# PATHWAY ENRICHMENT
# ==============================================================================

def perform_pathway_enrichment(reaction_stats):
    """Perform hypergeometric enrichment analysis"""
    print("\n" + "="*80)
    print("PATHWAY ENRICHMENT ANALYSIS")
    print("="*80)
    
    enrichment_results = {}
    
    for contrast, stats_df in reaction_stats.items():
        print(f"\nEnrichment for {contrast}...")
        
        if len(stats_df) == 0:
            continue
        
        sig_reactions = stats_df[stats_df['Significant']]
        
        if len(sig_reactions) == 0:
            print("  No significant reactions for enrichment")
            continue
        
        M = len(stats_df)  # Total reactions
        N = len(sig_reactions)  # Significant reactions
        
        enrichment_data = []
        
        for subsystem in stats_df['Subsystem'].unique():
            n = len(stats_df[stats_df['Subsystem'] == subsystem])  # Pathway size
            k = len(sig_reactions[sig_reactions['Subsystem'] == subsystem])  # Sig in pathway
            
            if k == 0:
                continue
            
            # Hypergeometric test: P(X >= k)
            p_value = hypergeom.sf(k - 1, M, n, N)
            
            # Enrichment ratio
            expected = (n * N) / M
            enrichment_ratio = k / expected if expected > 0 else 0
            
            enrichment_data.append({
                'Subsystem': subsystem,
                'k_significant': k,
                'n_total': n,
                'expected': expected,
                'enrichment_ratio': enrichment_ratio,
                'p_value': p_value
            })
        
        enrichment_df = pd.DataFrame(enrichment_data)
        
        if len(enrichment_df) > 0:
            # FDR correction
            enrichment_df['q_value'] = benjamini_hochberg_fdr(enrichment_df['p_value'])
            enrichment_df['Significant'] = enrichment_df['q_value'] < FDR_THRESHOLD
            enrichment_df = enrichment_df.sort_values('enrichment_ratio', ascending=False)
            
            n_enriched = enrichment_df['Significant'].sum()
            print(f"  Total pathways tested: {len(enrichment_df)}")
            print(f"  Significantly enriched: {n_enriched}")
            
            if n_enriched > 0:
                print(f"  Top 3 enriched:")
                for _, row in enrichment_df[enrichment_df['Significant']].head(3).iterrows():
                    print(f"    {row['Subsystem']}: {row['enrichment_ratio']:.2f}x "
                          f"({row['k_significant']}/{row['n_total']})")
        
        enrichment_results[contrast] = enrichment_df
    
    return enrichment_results

# ==============================================================================
# RANK-PRODUCT ANALYSIS
# ==============================================================================

def calculate_rank_product(reaction_stats, metadata_df):
    """Calculate rank-product statistic across datasets"""
    print("\n" + "="*80)
    print("RANK-PRODUCT ANALYSIS")
    print("="*80)
    
    rank_product_results = {}
    
    # Get contrasts that exist in multiple datasets
    unique_groups = sorted(metadata_df['Group'].unique())
    datasets = metadata_df['Dataset'].unique()
    
    # Use generate_pairwise_comparisons to ensure SCD is control (FIXED)
    comparisons_list = generate_pairwise_comparisons(unique_groups, control_group='SCD')
    
    for g1, g2 in comparisons_list:
        contrast_name = f"{g1}_vs_{g2}"
        
        # Check which datasets have both groups
        valid_datasets = []
        for dataset in datasets:
            dataset_groups = metadata_df[metadata_df['Dataset'] == dataset]['Group'].unique()
            if g1 in dataset_groups and g2 in dataset_groups:
                valid_datasets.append(dataset)
        
        if len(valid_datasets) < 2:
            print(f"\nSkipping {contrast_name}: only {len(valid_datasets)} dataset(s) with both groups")
            continue
        
        print(f"\n{contrast_name} in datasets: {', '.join(valid_datasets)}")
        
        # For rank-product, we'd need dataset-specific statistics
        # This is a simplified version using overall statistics
        # In practice, you'd calculate separate t-tests per dataset
        
        stats_df = reaction_stats[contrast_name].copy()
        
        if len(stats_df) > 0:
            # Rank by p-value
            stats_df['Rank'] = stats_df['p_value'].rank(method='min')
            stats_df['Rank_normalized'] = stats_df['Rank'] / len(stats_df)
            
            # Simple rank-product (would be geometric mean across datasets in full implementation)
            stats_df['RankProduct'] = stats_df['Rank_normalized']
            stats_df['RP_pvalue'] = stats_df['p_value']  # Placeholder
            
            print(f"  Top 5 reactions by rank-product:")
            for _, row in stats_df.head(5).iterrows():
                print(f"    {row['ReactionID']}: p={row['p_value']:.2e}, rank={row['Rank']}")
        
        rank_product_results[contrast_name] = stats_df
    
    return rank_product_results

# ==============================================================================
# NETWORK ANALYSIS
# ==============================================================================

def build_metabolic_network(reaction_stats, enrichment_results):
    """Build reaction-pathway network"""
    print("\n" + "="*80)
    print("NETWORK ANALYSIS")
    print("="*80)
    
    network_data = {}
    
    for contrast in reaction_stats.keys():
        print(f"\nBuilding network for {contrast}...")
        
        stats_df = reaction_stats[contrast]
        sig_reactions = stats_df[stats_df['Significant']]
        
        if len(sig_reactions) == 0:
            print("  No significant reactions")
            continue
        
        # Create network
        G = nx.Graph()
        
        # Add reaction nodes
        for _, row in sig_reactions.iterrows():
            G.add_node(row['ReactionID'],
                      type='reaction',
                      subsystem=row['Subsystem'],
                      effect=row['Cohen_d'],
                      pvalue=row['q_value'])
        
        # Add subsystem nodes and edges
        for subsystem in sig_reactions['Subsystem'].unique():
            G.add_node(subsystem, type='subsystem')
            
            subsys_reactions = sig_reactions[sig_reactions['Subsystem'] == subsystem]
            for rxn in subsys_reactions['ReactionID']:
                G.add_edge(rxn, subsystem)
        
        # Calculate network metrics
        metrics = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_reactions': len([n for n, d in G.nodes(data=True) if d.get('type') == 'reaction']),
            'n_subsystems': len([n for n, d in G.nodes(data=True) if d.get('type') == 'subsystem']),
            'density': nx.density(G),
            'n_components': nx.number_connected_components(G)
        }
        
        print(f"  Nodes: {metrics['n_nodes']} ({metrics['n_reactions']} reactions, {metrics['n_subsystems']} pathways)")
        print(f"  Edges: {metrics['n_edges']}")
        print(f"  Density: {metrics['density']:.3f}")
        
        # Identify hubs
        if len(G.nodes()) > 0:
            degree_centrality = nx.degree_centrality(G)
            top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            print(f"  Top hubs:")
            for node, centrality in top_hubs[:5]:
                node_type = G.nodes[node].get('type', 'unknown')
                print(f"    {node} ({node_type}): {centrality:.3f}")
        
        network_data[contrast] = {
            'graph': G,
            'metrics': metrics,
            'centrality': degree_centrality if len(G.nodes()) > 0 else {}
        }
    
    return network_data

def create_cytoscape_files(reaction_stats, subdirs):
    """Create Cytoscape-compatible network files"""
    print("\n" + "="*80)
    print("CREATING CYTOSCAPE FILES")
    print("="*80)
    
    for contrast, stats_df in reaction_stats.items():
        sig_reactions = stats_df[stats_df['Significant']]
        
        if len(sig_reactions) == 0:
            continue
        
        print(f"\nCreating files for {contrast}...")
        
        # Edge list: reaction → subsystem
        edge_data = []
        for _, row in sig_reactions.iterrows():
            edge_data.append({
                'source': row['ReactionID'],
                'target': row['Subsystem'],
                'interaction': 'in_pathway',
                'effect_size': row['Cohen_d'],
                'pvalue': row['p_value'],
                'qvalue': row['q_value']
            })
        
        edge_df = pd.DataFrame(edge_data)
        
        # Node attributes
        reaction_nodes = sig_reactions[['ReactionID', 'ReactionName', 'Subsystem', 'Cohen_d', 'q_value']].copy()
        reaction_nodes.columns = ['id', 'label', 'subsystem', 'effect_size', 'qvalue']
        reaction_nodes['type'] = 'reaction'
        
        subsystem_nodes = pd.DataFrame({
            'id': sig_reactions['Subsystem'].unique(),
            'label': sig_reactions['Subsystem'].unique(),
            'subsystem': sig_reactions['Subsystem'].unique(),
            'type': 'subsystem'
        })
        
        node_df = pd.concat([reaction_nodes, subsystem_nodes], ignore_index=True)
        
        # Save files
        edge_file = subdirs['networks'] / f"edges_{contrast}.csv"
        node_file = subdirs['networks'] / f"nodes_{contrast}.csv"
        
        edge_df.to_csv(edge_file, index=False)
        node_df.to_csv(node_file, index=False)
        
        print(f"  Saved {len(edge_df)} edges and {len(node_df)} nodes")

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_pca(pca_results, metadata_df, subdirs):
    """Plot PCA with confidence ellipses"""
    print("\nPlotting PCA...")
    
    scores = pca_results['scores']
    var_explained = pca_results['var_explained']
    groups = metadata_df['Group'].values
    unique_groups = sorted(metadata_df['Group'].unique())
    
    # Color palette
    colors = sns.color_palette("husl", len(unique_groups))
    color_map = dict(zip(unique_groups, colors))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA scatter
    ax = axes[0]
    for i, group in enumerate(unique_groups):
        mask = groups == group
        ax.scatter(scores[mask, 0], scores[mask, 1],
                  c=[color_map[group]], label=group,
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Confidence ellipse
        if mask.sum() > 2:
            calculate_confidence_ellipse(scores[mask, 0], scores[mask, 1],
                                        ax, n_std=1.96,
                                        edgecolor=color_map[group],
                                        facecolor='none', linewidth=2,
                                        linestyle='--', alpha=0.5)
    
    # Plot centroids
    for group in unique_groups:
        centroid = pca_results['centroids'][group]
        ax.plot(centroid[0], centroid[1], 'D',
               color=color_map[group], markersize=15,
               markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12, fontweight='bold')
    ax.set_title('PCA: Metabolic Flux Profiles', fontsize=14, fontweight='bold')
    ax.legend(title='Diet Group', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Scree plot
    ax = axes[1]
    n_components = min(20, len(var_explained))
    x = np.arange(1, n_components + 1)
    
    ax.plot(x, var_explained[:n_components] * 100, 'o-', linewidth=2, markersize=8, label='Individual')
    ax.plot(x, pca_results['cum_var'][:n_components] * 100, 's-', linewidth=2, markersize=8, label='Cumulative')
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax.set_title('Scree Plot', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'Figure_1_PCA_Analysis', subdirs)

def plot_volcano(reaction_stats, subdirs):
    """Create volcano plots for all contrasts"""
    print("\nPlotting volcano plots...")
    
    n_contrasts = len(reaction_stats)
    ncols = 3
    nrows = int(np.ceil(n_contrasts / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    # Always flatten when using multiple columns (even with single row)
    if nrows == 1 and ncols > 1:
        axes = axes.flatten()
    elif nrows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Single subplot case
    
    for idx, (contrast, stats_df) in enumerate(reaction_stats.items()):
        ax = axes[idx]
        
        if len(stats_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(contrast)
            continue
        
        # Prepare data
        x = stats_df['Cohen_d'].values
        y = -np.log10(stats_df['q_value'].values + 1e-300)
        
        significant = stats_df['Significant'].values
        
        # Plot non-significant
        ax.scatter(x[~significant], y[~significant],
                  c='gray', s=30, alpha=0.5, label='Not significant')
        
        # Plot significant
        if significant.sum() > 0:
            up = significant & (stats_df['Cohen_d'] > 0)
            down = significant & (stats_df['Cohen_d'] < 0)
            
            if up.sum() > 0:
                ax.scatter(x[up], y[up],
                          c='red', s=50, alpha=0.7, label=f'Up ({up.sum()})')
            
            if down.sum() > 0:
                ax.scatter(x[down], y[down],
                          c='blue', s=50, alpha=0.7, label=f'Down ({down.sum()})')
        
        # Threshold lines
        ax.axhline(y=-np.log10(FDR_THRESHOLD), color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=EFFECT_SIZE_THRESHOLD, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=-EFFECT_SIZE_THRESHOLD, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Labels
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=11, fontweight='bold')
        ax.set_ylabel('-log₁₀(q-value)', fontsize=11, fontweight='bold')
        ax.set_title(contrast.replace('_vs_', ' vs '), fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_contrasts, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'Figure_2_Volcano_Plots', subdirs)

def plot_heatmaps(df, metadata_df, sample_cols, reaction_stats, subdirs):
    """Create heatmaps"""
    print("\nPlotting heatmaps...")
    
    # Get top variable reactions across all contrasts
    all_sig_reactions = set()
    for stats_df in reaction_stats.values():
        sig = stats_df[stats_df['Significant']]['ReactionID'].tolist()
        all_sig_reactions.update(sig)
    
    if len(all_sig_reactions) == 0:
        print("  No significant reactions for heatmap")
        return
    
    # Limit to top 50 by variance
    flux_data = df[sample_cols].loc[df['ReactionID'].isin(all_sig_reactions)]
    flux_data.index = df.loc[df['ReactionID'].isin(all_sig_reactions), 'ReactionID']
    
    if len(flux_data) > 50:
        variances = flux_data.var(axis=1)
        top_reactions = variances.nlargest(50).index
        flux_data = flux_data.loc[top_reactions]
    
    # Z-score normalization
    flux_zscore = (flux_data.T - flux_data.mean(axis=1)) / flux_data.std(axis=1)
    flux_zscore = flux_zscore.T
    
    # Create color map for groups
    unique_groups = sorted(metadata_df['Group'].unique())
    group_colors = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))
    col_colors = [group_colors[g] for g in metadata_df['Group']]
    
    # Plot
    fig = plt.figure(figsize=(14, 10))
    
    g = sns.clustermap(flux_zscore,
                      cmap='RdBu_r',
                      center=0,
                      col_colors=col_colors,
                      cbar_kws={'label': 'Z-score'},
                      figsize=(14, 10),
                      dendrogram_ratio=0.15,
                      yticklabels=True,
                      xticklabels=False)
    
    g.ax_heatmap.set_xlabel('Samples', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('Reactions', fontsize=12, fontweight='bold')
    
    # Add legend
    handles = [mpatches.Patch(color=color, label=group) 
               for group, color in group_colors.items()]
    g.ax_heatmap.legend(handles=handles, title='Group',
                       bbox_to_anchor=(1.3, 1), loc='upper left',
                       frameon=True)
    
    plt.suptitle('Top Significant Reactions (Hierarchical Clustering)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    save_figure(g.fig, 'Figure_3_Reaction_Heatmap', subdirs)

def plot_subsystem_enrichment(enrichment_results, subdirs):
    """Plot subsystem enrichment"""
    print("\nPlotting subsystem enrichment...")
    
    n_contrasts = len(enrichment_results)
    if n_contrasts == 0:
        return
    
    ncols = 2
    nrows = int(np.ceil(n_contrasts / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    # Always flatten when using multiple columns (even with single row)
    if nrows == 1 and ncols > 1:
        axes = axes.flatten()
    elif nrows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Single subplot case
    
    for idx, (contrast, enrich_df) in enumerate(enrichment_results.items()):
        ax = axes[idx]
        
        if len(enrich_df) == 0 or enrich_df['Significant'].sum() == 0:
            ax.text(0.5, 0.5, 'No enrichment', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(contrast)
            continue
        
        # Plot top enriched pathways
        plot_df = enrich_df[enrich_df['Significant']].head(15).copy()
        plot_df = plot_df.sort_values('enrichment_ratio')
        
        colors = ['red' if r > 1 else 'blue' for r in plot_df['enrichment_ratio']]
        
        y_pos = np.arange(len(plot_df))
        ax.barh(y_pos, plot_df['enrichment_ratio'], color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['Subsystem'], fontsize=9)
        ax.set_xlabel('Enrichment Ratio', fontsize=11, fontweight='bold')
        ax.set_title(contrast.replace('_vs_', ' vs '), fontsize=12, fontweight='bold')
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
    
    # Hide unused subplots
    for idx in range(n_contrasts, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'Figure_4_Subsystem_Enrichment', subdirs)

def plot_pathway_direction(subsystem_results, subdirs):
    """Plot pathway direction changes"""
    print("\nPlotting pathway direction analysis...")
    
    n_contrasts = len(subsystem_results)
    if n_contrasts == 0:
        return
    
    ncols = 2
    nrows = int(np.ceil(n_contrasts / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    # Always flatten when using multiple columns (even with single row)
    if nrows == 1 and ncols > 1:
        axes = axes.flatten()
    elif nrows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Single subplot case
    
    for idx, (contrast, subsys_df) in enumerate(subsystem_results.items()):
        ax = axes[idx]
        
        if len(subsys_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(contrast)
            continue
        
        # Get top pathways by effect size
        plot_df = subsys_df.nlargest(15, 'Mean_abs_Cohen_d').copy()
        plot_df = plot_df.sort_values('Mean_Cohen_d')
        
        # Create stacked bar for up/down
        y_pos = np.arange(len(plot_df))
        
        up_counts = plot_df['N_positive'].values
        down_counts = plot_df['N_negative'].values
        
        ax.barh(y_pos, up_counts, color='red', alpha=0.7, label='Up')
        ax.barh(y_pos, -down_counts, color='blue', alpha=0.7, label='Down')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['Subsystem'], fontsize=9)
        ax.set_xlabel('Number of Reactions', fontsize=11, fontweight='bold')
        ax.set_title(contrast.replace('_vs_', ' vs '), fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
    
    # Hide unused subplots
    for idx in range(n_contrasts, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'Figure_5_Pathway_Direction', subdirs)

def plot_effect_size_distributions(reaction_stats, subdirs):
    """Plot effect size distributions"""
    print("\nPlotting effect size distributions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collect all effect sizes
    all_effects = []
    sig_effects = []
    
    for stats_df in reaction_stats.values():
        all_effects.extend(stats_df['Cohen_d'].values)
        sig_effects.extend(stats_df[stats_df['Significant']]['Cohen_d'].values)
    
    all_effects = np.array(all_effects)
    sig_effects = np.array(sig_effects)
    
    # Distribution of all effects
    ax = axes[0, 0]
    ax.hist(all_effects, bins=50, color='gray', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=2)
    ax.axvline(x=EFFECT_SIZE_THRESHOLD, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.axvline(x=-EFFECT_SIZE_THRESHOLD, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Cohen's d", fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('All Effect Sizes', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution of significant effects
    ax = axes[0, 1]
    if len(sig_effects) > 0:
        ax.hist(sig_effects, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linewidth=2)
    else:
        ax.text(0.5, 0.5, 'No significant effects', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel("Cohen's d", fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Significant Effect Sizes', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Box plot by contrast
    ax = axes[1, 0]
    contrast_names = list(reaction_stats.keys())
    contrast_effects = [reaction_stats[c]['Cohen_d'].values for c in contrast_names]
    
    bp = ax.boxplot(contrast_effects, labels=contrast_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=EFFECT_SIZE_THRESHOLD, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-EFFECT_SIZE_THRESHOLD, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel("Cohen's d", fontsize=12, fontweight='bold')
    ax.set_title('Effect Sizes by Contrast', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary statistics
    ax = axes[1, 1]
    summary_data = []
    for contrast, stats_df in reaction_stats.items():
        summary_data.append({
            'Contrast': contrast.replace('_vs_', '\nvs\n'),
            'Mean |d|': stats_df['Cohen_d'].abs().mean(),
            'Median |d|': stats_df['Cohen_d'].abs().median(),
            'Max |d|': stats_df['Cohen_d'].abs().max()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    x = np.arange(len(summary_df))
    width = 0.25
    
    ax.bar(x - width, summary_df['Mean |d|'], width, label='Mean |d|', alpha=0.8)
    ax.bar(x, summary_df['Median |d|'], width, label='Median |d|', alpha=0.8)
    ax.bar(x + width, summary_df['Max |d|'], width, label='Max |d|', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Contrast'], fontsize=8)
    ax.set_ylabel('Effect Size', fontsize=12, fontweight='bold')
    ax.set_title('Effect Size Summary', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'Figure_6_Effect_Size_Distributions', subdirs)

def plot_network_visualization(network_data, reaction_stats, subdirs):
    """Plot network visualizations"""
    print("\nPlotting network visualizations...")
    
    n_contrasts = len(network_data)
    if n_contrasts == 0:
        return
    
    ncols = 2
    nrows = int(np.ceil(n_contrasts / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8 * nrows))
    # Always flatten when using multiple columns (even with single row)
    if nrows == 1 and ncols > 1:
        axes = axes.flatten()
    elif nrows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Single subplot case
    
    for idx, (contrast, net_info) in enumerate(network_data.items()):
        ax = axes[idx]
        G = net_info['graph']
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'No network', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(contrast)
            continue
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=RANDOM_SEED)
        
        # Separate nodes by type
        reaction_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'reaction']
        subsystem_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'subsystem']
        
        # Node colors and sizes
        reaction_effects = [G.nodes[n].get('effect', 0) for n in reaction_nodes]
        reaction_colors = ['red' if e > 0 else 'blue' for e in reaction_effects]
        reaction_sizes = [abs(e) * 300 + 100 for e in reaction_effects]
        
        # Draw
        nx.draw_networkx_nodes(G, pos, nodelist=reaction_nodes,
                              node_color=reaction_colors, node_size=reaction_sizes,
                              alpha=0.7, ax=ax)
        
        nx.draw_networkx_nodes(G, pos, nodelist=subsystem_nodes,
                              node_color='gold', node_size=500,
                              node_shape='s', alpha=0.8, ax=ax)
        
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        
        # Labels for subsystems only
        subsystem_labels = {n: n for n in subsystem_nodes}
        nx.draw_networkx_labels(G, pos, subsystem_labels, font_size=8, ax=ax)
        
        ax.set_title(f"{contrast.replace('_vs_', ' vs ')}\n"
                    f"{net_info['metrics']['n_reactions']} reactions, "
                    f"{net_info['metrics']['n_subsystems']} pathways",
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_contrasts, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'Figure_7_Network_Visualization', subdirs)

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_comprehensive_report(output_dir, subdirs, results):
    """Generate comprehensive analysis report"""
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    report_path = subdirs['reports'] / 'COMPREHENSIVE_ANALYSIS_REPORT.md'
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Metabolic Flux Analysis Report\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset summary
        f.write("## Dataset Summary\n\n")
        f.write(f"- **Total Reactions**: {results['n_reactions']}\n")
        f.write(f"- **Total Samples**: {results['n_samples']}\n")
        f.write(f"- **Diet Groups**: {', '.join(results['groups'])}\n")
        f.write(f"- **Datasets**: {', '.join(results['datasets'])}\n\n")
        
        # PCA results
        f.write("## PCA Analysis\n\n")
        f.write(f"- **PC1 Variance**: {results['pca']['var1']:.2%}\n")
        f.write(f"- **PC2 Variance**: {results['pca']['var2']:.2%}\n")
        f.write(f"- **Cumulative Variance (PC1+PC2)**: {results['pca']['cum_var']:.2%}\n\n")
        
        f.write("### Centroid Distances (PC1-PC2 space)\n\n")
        f.write("| Comparison | Distance |\n")
        f.write("|------------|----------|\n")
        for contrast, dist in results['pca']['distances'].items():
            f.write(f"| {contrast} | {dist:.3f} |\n")
        f.write("\n")
        
        # PERMANOVA
        f.write("## PERMANOVA Results\n\n")
        f.write("### Overall\n\n")
        f.write(f"- **F-statistic**: {results['permanova']['overall']['F']:.3f}\n")
        f.write(f"- **p-value**: {results['permanova']['overall']['p']:.4f}\n")
        f.write(f"- **R²**: {results['permanova']['overall']['R2']:.3f}\n\n")
        
        f.write("### Pairwise Comparisons\n\n")
        f.write("| Contrast | F-statistic | p-value | q-value | R² |\n")
        f.write("|----------|-------------|---------|---------|----|\n")
        for _, row in results['permanova']['pairwise'].iterrows():
            f.write(f"| {row['Contrast']} | {row['F']:.3f} | {row['p']:.4f} | "
                   f"{row['q_value']:.4f} | {row['R2']:.3f} |\n")
        f.write("\n")
        
        # Reaction statistics
        f.write("## Reaction-Level Differential Analysis\n\n")
        for contrast, stats in results['reaction_summary'].items():
            f.write(f"### {contrast}\n\n")
            f.write(f"- **Total reactions tested**: {stats['total']}\n")
            f.write(f"- **Significant reactions**: {stats['significant']} ({stats['pct_sig']:.1%})\n")
            f.write(f"- **Upregulated**: {stats['up']}\n")
            f.write(f"- **Downregulated**: {stats['down']}\n")
            f.write(f"- **Mean |Cohen's d|**: {stats['mean_d']:.3f}\n")
            f.write(f"- **Median |Cohen's d|**: {stats['median_d']:.3f}\n\n")
        
        # Subsystem analysis
        f.write("## Subsystem/Pathway Analysis\n\n")
        for contrast, subsys_stats in results['subsystem_summary'].items():
            f.write(f"### {contrast}\n\n")
            f.write(f"- **Total subsystems**: {subsys_stats['total']}\n")
            f.write(f"- **Sign-consistent subsystems**: {subsys_stats['sign_consistent']}\n\n")
            
            if subsys_stats['top_pathways']:
                f.write("**Top 5 Pathways by Effect Size**:\n\n")
                for i, pw in enumerate(subsys_stats['top_pathways'], 1):
                    f.write(f"{i}. {pw['name']}: d={pw['effect']:.3f}, "
                           f"{pw['n_sig']}/{pw['n_total']} significant\n")
                f.write("\n")
        
        # Enrichment analysis
        f.write("## Pathway Enrichment Analysis\n\n")
        for contrast, enrich_stats in results['enrichment_summary'].items():
            f.write(f"### {contrast}\n\n")
            f.write(f"- **Pathways tested**: {enrich_stats['total']}\n")
            f.write(f"- **Significantly enriched**: {enrich_stats['enriched']}\n\n")
            
            if enrich_stats['top_enriched']:
                f.write("**Top 5 Enriched Pathways**:\n\n")
                for i, pw in enumerate(enrich_stats['top_enriched'], 1):
                    f.write(f"{i}. {pw['name']}: {pw['ratio']:.2f}x enrichment "
                           f"({pw['k']}/{pw['n']} reactions, q={pw['qval']:.2e})\n")
                f.write("\n")
        
        # Network analysis
        f.write("## Network Analysis\n\n")
        for contrast, net_stats in results['network_summary'].items():
            f.write(f"### {contrast}\n\n")
            f.write(f"- **Nodes**: {net_stats['nodes']} ({net_stats['reactions']} reactions, "
                   f"{net_stats['subsystems']} pathways)\n")
            f.write(f"- **Edges**: {net_stats['edges']}\n")
            f.write(f"- **Density**: {net_stats['density']:.3f}\n")
            f.write(f"- **Connected components**: {net_stats['components']}\n\n")
        
        # File outputs
        f.write("## Generated Files\n\n")
        f.write("### CSV Files\n\n")
        csv_files = sorted(subdirs['csv'].glob('*.csv'))
        for csv_file in csv_files:
            f.write(f"- `{csv_file.name}`\n")
        
        f.write("\n### Figures\n\n")
        figure_files = sorted(subdirs['figures'].glob('*.png'))
        for fig_file in figure_files:
            f.write(f"- `{fig_file.name}`\n")
        
        f.write("\n### Network Files (Cytoscape)\n\n")
        network_files = sorted(subdirs['networks'].glob('*.csv'))
        for net_file in network_files:
            f.write(f"- `{net_file.name}`\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("\n**Analysis completed successfully!**\n")
    
    print(f"\nReport saved to: {report_path}")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    # Declare globals at the start
    global N_PERMUTATIONS, FDR_THRESHOLD, EFFECT_SIZE_THRESHOLD
    
    parser = argparse.ArgumentParser(
        description='Comprehensive metabolic flux analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python comprehensive_flux_analysis.py \\
      --input combined_reaction_flux_comparison_extended_batch_corrected_3d.csv \\
      --output Comprehensive_Flux_Analysis
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with flux data')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory')
    parser.add_argument('--permutations', '-p', type=int, default=N_PERMUTATIONS,
                       help=f'Number of permutations for PERMANOVA (default: {N_PERMUTATIONS})')
    parser.add_argument('--fdr', '-f', type=float, default=FDR_THRESHOLD,
                       help=f'FDR threshold (default: {FDR_THRESHOLD})')
    parser.add_argument('--effect-size', '-e', type=float, default=EFFECT_SIZE_THRESHOLD,
                       help=f'Effect size threshold (default: {EFFECT_SIZE_THRESHOLD})')
    
    args = parser.parse_args()
    
    # Update settings from command line
    N_PERMUTATIONS = args.permutations
    FDR_THRESHOLD = args.fdr
    EFFECT_SIZE_THRESHOLD = args.effect_size
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE METABOLIC FLUX ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"\nInput file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Permutations: {N_PERMUTATIONS}")
    print(f"FDR threshold: {FDR_THRESHOLD}")
    print(f"Effect size threshold: {EFFECT_SIZE_THRESHOLD}")
    
    # Setup output directory
    output_dir, subdirs = setup_output_dir(args.output)
    
    # Load data
    df, annotations, flux_data, metadata_df, sample_cols = load_and_prepare_data(args.input)
    
    # Save metadata
    metadata_df.to_csv(subdirs['csv'] / 'sample_metadata.csv', index=False)
    
    # PCA analysis
    pca_results = perform_pca(flux_data, metadata_df)
    
    # Save PCA results (use actual number of components, not hardcoded 10)
    n_components = pca_results['scores'].shape[1]
    pca_scores_df = pd.DataFrame(pca_results['scores'],
                                 columns=[f'PC{i+1}' for i in range(n_components)])
    pca_scores_df['SampleID'] = metadata_df['SampleID'].values
    pca_scores_df['Group'] = metadata_df['Group'].values
    pca_scores_df['Dataset'] = metadata_df['Dataset'].values
    pca_scores_df.to_csv(subdirs['csv'] / 'pca_scores.csv', index=False)
    
    # Save variance explained
    var_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(pca_results['var_explained']))],
        'Variance': pca_results['var_explained'],
        'Cumulative_Variance': pca_results['cum_var']
    })
    var_df.to_csv(subdirs['csv'] / 'pca_variance_explained.csv', index=False)
    
    # Save centroid distances
    centroid_dist_df = pd.DataFrame(list(pca_results['centroid_distances'].items()),
                                    columns=['Contrast', 'Distance'])
    centroid_dist_df.to_csv(subdirs['csv'] / 'centroid_distances_PC12.csv', index=False)
    
    # PERMANOVA
    overall_permanova, pairwise_permanova = perform_permanova_analysis(pca_results, metadata_df)
    
    # Save PERMANOVA results
    pd.DataFrame([overall_permanova]).to_csv(subdirs['csv'] / 'permanova_overall.csv', index=False)
    pairwise_permanova.to_csv(subdirs['csv'] / 'permanova_pairwise.csv', index=False)
    
    # Reaction-level statistics
    reaction_stats = calculate_reaction_statistics(df, metadata_df, sample_cols, annotations)
    
    # Save reaction statistics
    for contrast, stats_df in reaction_stats.items():
        stats_df.to_csv(subdirs['csv'] / f'reaction_stats_{contrast}.csv', index=False)
    
    # Subsystem analysis
    subsystem_results = analyze_subsystems(reaction_stats)
    
    # Save subsystem results
    for contrast, subsys_df in subsystem_results.items():
        subsys_df.to_csv(subdirs['csv'] / f'subsystem_analysis_{contrast}.csv', index=False)
    
    # Pathway enrichment
    enrichment_results = perform_pathway_enrichment(reaction_stats)
    
    # Save enrichment results
    for contrast, enrich_df in enrichment_results.items():
        if len(enrich_df) > 0:
            enrich_df.to_csv(subdirs['csv'] / f'pathway_enrichment_{contrast}.csv', index=False)
    
    # Rank-product analysis
    rank_product_results = calculate_rank_product(reaction_stats, metadata_df)
    
    # Save rank-product results
    for contrast, rp_df in rank_product_results.items():
        if len(rp_df) > 0:
            rp_df.to_csv(subdirs['csv'] / f'rank_product_{contrast}.csv', index=False)
    
    # Network analysis
    network_data = build_metabolic_network(reaction_stats, enrichment_results)
    
    # Save network metrics
    network_metrics = []
    for contrast, net_info in network_data.items():
        metrics = net_info['metrics'].copy()
        metrics['Contrast'] = contrast
        network_metrics.append(metrics)
    
    if network_metrics:
        pd.DataFrame(network_metrics).to_csv(subdirs['csv'] / 'network_metrics.csv', index=False)
    
    # Create Cytoscape files
    create_cytoscape_files(reaction_stats, subdirs)
    
    # Visualizations
    plot_pca(pca_results, metadata_df, subdirs)
    plot_volcano(reaction_stats, subdirs)
    plot_heatmaps(df, metadata_df, sample_cols, reaction_stats, subdirs)
    plot_subsystem_enrichment(enrichment_results, subdirs)
    plot_pathway_direction(subsystem_results, subdirs)
    plot_effect_size_distributions(reaction_stats, subdirs)
    plot_network_visualization(network_data, reaction_stats, subdirs)
    
    # Prepare summary for report
    results = {
        'n_reactions': len(df),
        'n_samples': len(metadata_df),
        'groups': sorted(metadata_df['Group'].unique().tolist()),
        'datasets': sorted(metadata_df['Dataset'].unique().tolist()),
        'pca': {
            'var1': pca_results['var_explained'][0],
            'var2': pca_results['var_explained'][1],
            'cum_var': pca_results['cum_var'][1],
            'distances': pca_results['centroid_distances']
        },
        'permanova': {
            'overall': overall_permanova,
            'pairwise': pairwise_permanova
        },
        'reaction_summary': {},
        'subsystem_summary': {},
        'enrichment_summary': {},
        'network_summary': {}
    }
    
    # Compile reaction summaries
    for contrast, stats_df in reaction_stats.items():
        n_sig = stats_df['Significant'].sum()
        results['reaction_summary'][contrast] = {
            'total': len(stats_df),
            'significant': int(n_sig),
            'pct_sig': n_sig / len(stats_df) if len(stats_df) > 0 else 0,
            'up': int((stats_df['Significant'] & (stats_df['MeanDiff'] > 0)).sum()),
            'down': int((stats_df['Significant'] & (stats_df['MeanDiff'] < 0)).sum()),
            'mean_d': float(stats_df['Cohen_d'].abs().mean()),
            'median_d': float(stats_df['Cohen_d'].abs().median())
        }
    
    # Compile subsystem summaries
    for contrast, subsys_df in subsystem_results.items():
        top_pathways = []
        if len(subsys_df) > 0:
            for _, row in subsys_df.head(5).iterrows():
                top_pathways.append({
                    'name': row['Subsystem'],
                    'effect': float(row['Mean_Cohen_d']),
                    'n_sig': int(row['N_significant']),
                    'n_total': int(row['N_reactions'])
                })
        
        results['subsystem_summary'][contrast] = {
            'total': len(subsys_df),
            'sign_consistent': int(subsys_df['Sign_consistent'].sum()) if len(subsys_df) > 0 else 0,
            'top_pathways': top_pathways
        }
    
    # Compile enrichment summaries
    for contrast, enrich_df in enrichment_results.items():
        top_enriched = []
        if len(enrich_df) > 0 and enrich_df['Significant'].sum() > 0:
            for _, row in enrich_df[enrich_df['Significant']].head(5).iterrows():
                top_enriched.append({
                    'name': row['Subsystem'],
                    'ratio': float(row['enrichment_ratio']),
                    'k': int(row['k_significant']),
                    'n': int(row['n_total']),
                    'qval': float(row['q_value'])
                })
        
        results['enrichment_summary'][contrast] = {
            'total': len(enrich_df),
            'enriched': int(enrich_df['Significant'].sum()) if len(enrich_df) > 0 else 0,
            'top_enriched': top_enriched
        }
    
    # Compile network summaries
    for contrast, net_info in network_data.items():
        results['network_summary'][contrast] = {
            'nodes': net_info['metrics']['n_nodes'],
            'reactions': net_info['metrics']['n_reactions'],
            'subsystems': net_info['metrics']['n_subsystems'],
            'edges': net_info['metrics']['n_edges'],
            'density': net_info['metrics']['density'],
            'components': net_info['metrics']['n_components']
        }
    
    # Generate report
    generate_comprehensive_report(output_dir, subdirs, results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"\nKey outputs:")
    print(f"  - CSV files: {subdirs['csv']}")
    print(f"  - Figures: {subdirs['figures']}")
    print(f"  - Cytoscape networks: {subdirs['networks']}")
    print(f"  - Report: {subdirs['reports'] / 'COMPREHENSIVE_ANALYSIS_REPORT.md'}")
    print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    main()
