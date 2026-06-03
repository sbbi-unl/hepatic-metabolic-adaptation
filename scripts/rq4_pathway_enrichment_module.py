#!/usr/bin/env python3
"""
RQ4: Pathway-Level Analysis Module
===================================
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


class RQ4PathwayEnrichment:
    """
    Comprehensive pathway and compartment enrichment analysis.
    
    This class integrates with the RQ4 attribution analysis to provide
    pathway-level insights into microbiome-host metabolic interactions.
    """
    
    def __init__(self, model_file: str, attribution_csv: str, output_dir: str):
        """
        Initialize pathway enrichment analysis.
        
        Parameters
        ----------
        model_file : str
            Path to iMM1415.json metabolic model
        attribution_csv : str
            Path to flux_attribution_analysis.csv from Stage 3
        output_dir : str
            Directory for saving results
        """
        self.model_file = model_file
        self.attribution_csv = attribution_csv
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        print("="*80)
        print("RQ4: PATHWAY ENRICHMENT ANALYSIS")
        print("="*80)
        
        self.load_model_annotations()
        self.load_attribution_data()
        
    def load_model_annotations(self):
        """
        Extract pathway (subsystem) and compartment annotations from iMM1415.
        
        Creates:
        --------
        self.reaction_annotations : dict
            {reaction_id: {'subsystem': str, 'compartment': str, 'name': str}}
        self.all_subsystems : set
            All unique pathway names
        self.all_compartments : set
            All unique compartment codes
        """
        print("\n[INFO] Loading iMM1415 model annotations...")
        
        with open(self.model_file, 'r') as f:
            model_data = json.load(f)
        
        reactions = model_data.get('reactions', [])
        print(f"  Loaded {len(reactions)} reactions from model")
        
        # Extract annotations
        self.reaction_annotations = {}
        subsystems = set()
        compartments = set()
        
        for rxn in reactions:
            rxn_id = rxn['id']
            subsystem = rxn.get('subsystem', 'Unknown')
            name = rxn.get('name', rxn_id)
            
            # Infer compartment from reaction ID
            # iMM1415 uses suffixes: c (cytosol), m (mitochondria), r (ER), x (peroxisome), etc.
            compartment = self._infer_compartment(rxn_id)
            
            self.reaction_annotations[rxn_id] = {
                'subsystem': subsystem if subsystem else 'Unknown',
                'compartment': compartment,
                'name': name
            }
            
            if subsystem:
                subsystems.add(subsystem)
            compartments.add(compartment)
        
        self.all_subsystems = subsystems
        self.all_compartments = compartments
        
        print(f"  Found {len(subsystems)} unique pathways/subsystems")
        print(f"  Found {len(compartments)} unique compartments")
        print(f"  Annotated {len(self.reaction_annotations)} reactions")
        
        # Show sample subsystems
        print(f"\n  Sample pathways:")
        for s in sorted(list(subsystems))[:10]:
            count = sum(1 for r in self.reaction_annotations.values() if r['subsystem'] == s)
            print(f"    - {s}: {count} reactions")
    
    def _infer_compartment(self, reaction_id: str) -> str:
        """
        Infer cellular compartment from reaction ID suffix.
        
        iMM1415 convention:
        - c: cytosol
        - m: mitochondria
        - r: endoplasmic reticulum
        - x: peroxisome
        - l: lysosome
        - n: nucleus
        - e: extracellular
        - (no suffix): assumed cytosol
        """
        if reaction_id.endswith('m'):
            return 'mitochondria'
        elif reaction_id.endswith('r'):
            return 'ER'
        elif reaction_id.endswith('x'):
            return 'peroxisome'
        elif reaction_id.endswith('l'):
            return 'lysosome'
        elif reaction_id.endswith('n'):
            return 'nucleus'
        elif reaction_id.endswith('e'):
            return 'extracellular'
        elif reaction_id.endswith('c') or not any(reaction_id.endswith(s) for s in ['m', 'r', 'x', 'l', 'n', 'e']):
            return 'cytosol'
        else:
            return 'unknown'
    
    def load_attribution_data(self):
        """
        Load flux attribution analysis results.
        
        Creates:
        --------
        self.attribution_df : pd.DataFrame
            Flux attribution data with subsystem/compartment annotations
        """
        print(f"\n[INFO] Loading attribution data: {os.path.basename(self.attribution_csv)}")
        
        self.attribution_df = pd.read_csv(self.attribution_csv)
        print(f"  Loaded {len(self.attribution_df)} reactions")
        
        # Add annotations
        self.attribution_df['subsystem'] = self.attribution_df['reaction_id'].map(
            lambda x: self.reaction_annotations.get(x, {}).get('subsystem', 'Unknown')
        )
        self.attribution_df['compartment'] = self.attribution_df['reaction_id'].map(
            lambda x: self.reaction_annotations.get(x, {}).get('compartment', 'unknown')
        )
        self.attribution_df['reaction_name'] = self.attribution_df['reaction_id'].map(
            lambda x: self.reaction_annotations.get(x, {}).get('name', x)
        )
        
        # Filter to significant reactions
        self.significant_df = self.attribution_df[
            self.attribution_df['delta_total'].abs() > 1e-3
        ].copy()
        
        print(f"  Significant reactions: {len(self.significant_df)} ({100*len(self.significant_df)/len(self.attribution_df):.1f}%)")
        
        # Summary statistics
        subsystem_counts = self.significant_df['subsystem'].value_counts()
        print(f"\n  Top 5 pathways by reaction count:")
        for pathway, count in subsystem_counts.head(5).items():
            print(f"    - {pathway}: {count} reactions")
    
    def pathway_enrichment_analysis(self):
        """
        Perform pathway enrichment using hypergeometric test.
        
        For each pathway, tests whether significantly altered reactions
        are over-represented compared to background expectation.
        
        NULL HYPOTHESIS:
        ----------------
        Significantly altered reactions are randomly distributed across pathways.
        
        TEST STATISTIC:
        ---------------
        Hypergeometric probability:
        - k: # significant reactions in pathway
        - M: Total reactions analyzed
        - n: Total reactions in pathway
        - N: Total significant reactions
        
        P(X >= k) = hypergeom.sf(k-1, M, n, N)
        
        ENRICHMENT RATIO:
        -----------------
        (k/n) / (N/M) = observed frequency / expected frequency
        
        Returns:
        --------
        pd.DataFrame with columns:
        - subsystem: Pathway name
        - n_reactions: Total reactions in pathway
        - n_significant: Significant reactions in pathway
        - expected: Expected significant reactions
        - enrichment_ratio: Observed/Expected
        - p_value: Hypergeometric p-value
        - p_adjusted: FDR-corrected p-value
        - diet_dominated: % reactions where diet > microbiome
        - microbiome_dominated: % reactions where microbiome > diet
        - synergistic: % reactions with same-direction effects
        """
        print("\n" + "="*80)
        print("PATHWAY ENRICHMENT ANALYSIS")
        print("="*80)
        
        # Background: all reactions
        M = len(self.attribution_df)
        N = len(self.significant_df)
        
        print(f"\n[INFO] Hypergeometric test setup:")
        print(f"  Total reactions (M): {M}")
        print(f"  Total significant (N): {N}")
        print(f"  Background rate: {100*N/M:.1f}%")
        
        enrichment_results = []
        
        for subsystem in self.all_subsystems:
            if subsystem == 'Unknown':
                continue
            
            # Get reactions in this pathway
            pathway_reactions = self.attribution_df[
                self.attribution_df['subsystem'] == subsystem
            ]
            pathway_significant = self.significant_df[
                self.significant_df['subsystem'] == subsystem
            ]
            
            n = len(pathway_reactions)
            k = len(pathway_significant)
            
            if k == 0 or n < 2:  # Skip pathways with <2 reactions or 0 significant
                continue
            
            # Hypergeometric test
            expected = (n * N) / M
            p_value = hypergeom.sf(k - 1, M, n, N)
            enrichment_ratio = (k / expected) if expected > 0 else 0
            
            # Calculate directional statistics
            if k > 0:
                diet_dominated = 100 * sum(
                    pathway_significant['variance_explained_diet'] > 70
                ) / k
                
                microbiome_dominated = 100 * sum(
                    pathway_significant['variance_explained_microbiome'] > 70
                ) / k
                
                synergistic = 100 * sum(
                    pathway_significant['effect_type'] == 'Synergistic'
                ) / k
                
                mean_diet_variance = pathway_significant['variance_explained_diet'].mean()
                mean_mb_variance = pathway_significant['variance_explained_microbiome'].mean()
            else:
                diet_dominated = 0
                microbiome_dominated = 0
                synergistic = 0
                mean_diet_variance = 0
                mean_mb_variance = 0
            
            enrichment_results.append({
                'subsystem': subsystem,
                'n_reactions': n,
                'n_significant': k,
                'expected': expected,
                'enrichment_ratio': enrichment_ratio,
                'p_value': p_value,
                'diet_dominated_pct': diet_dominated,
                'microbiome_dominated_pct': microbiome_dominated,
                'synergistic_pct': synergistic,
                'mean_diet_variance': mean_diet_variance,
                'mean_microbiome_variance': mean_mb_variance
            })
        
        # Create DataFrame
        enrichment_df = pd.DataFrame(enrichment_results)
        
        # FDR correction
        enrichment_df['p_adjusted'] = multipletests(
            enrichment_df['p_value'], 
            method='fdr_bh'
        )[1]
        
        # Add significance flag
        enrichment_df['significant'] = enrichment_df['p_adjusted'] < 0.05
        
        # Sort by enrichment ratio
        enrichment_df = enrichment_df.sort_values('enrichment_ratio', ascending=False)
        
        # Save results
        output_file = os.path.join(self.output_dir, 'pathway_enrichment_results.csv')
        enrichment_df.to_csv(output_file, index=False)
        print(f"\n[SAVED] {output_file}")
        
        # Print summary
        sig_pathways = enrichment_df[enrichment_df['significant']]
        print(f"\n[RESULTS] Significantly enriched pathways: {len(sig_pathways)}/{len(enrichment_df)}")
        
        if len(sig_pathways) > 0:
            print(f"\n  Top 10 enriched pathways:")
            for idx, row in sig_pathways.head(10).iterrows():
                print(f"    {row['subsystem'][:50]:50s} {row['n_significant']:3d}/{row['n_reactions']:3d} "
                      f"(enrichment={row['enrichment_ratio']:.2f}x, p={row['p_adjusted']:.2e})")
        
        self.pathway_enrichment_df = enrichment_df
        return enrichment_df
    
    def compartment_enrichment_analysis(self):
        """
        Perform compartment enrichment analysis.
        
        Tests whether microbiome effects are concentrated in specific
        cellular compartments (mitochondria, cytosol, ER, peroxisome).
        
        This answers: "Where in the cell do microbiome effects concentrate?"
        
        Returns:
        --------
        pd.DataFrame with compartment enrichment statistics
        """
        print("\n" + "="*80)
        print("COMPARTMENT ENRICHMENT ANALYSIS")
        print("="*80)
        
        # Background
        M = len(self.attribution_df)
        N = len(self.significant_df)
        
        compartment_results = []
        
        for compartment in self.all_compartments:
            if compartment == 'unknown':
                continue
            
            # Get reactions in this compartment
            comp_reactions = self.attribution_df[
                self.attribution_df['compartment'] == compartment
            ]
            comp_significant = self.significant_df[
                self.significant_df['compartment'] == compartment
            ]
            
            n = len(comp_reactions)
            k = len(comp_significant)
            
            if k == 0 or n < 2:
                continue
            
            # Hypergeometric test
            expected = (n * N) / M
            p_value = hypergeom.sf(k - 1, M, n, N)
            enrichment_ratio = (k / expected) if expected > 0 else 0
            
            # Calculate microbiome contribution statistics
            if k > 0:
                mean_mb_contribution = comp_significant['abs_microbiome_contribution'].mean()
                mean_diet_contribution = comp_significant['abs_diet_contribution'].mean()
                mean_mb_variance = comp_significant['variance_explained_microbiome'].mean()
                
                # Large microbiome effects
                large_mb_effects = sum(comp_significant['microbiome_effect_size'] == 'Large')
                large_mb_pct = 100 * large_mb_effects / k
            else:
                mean_mb_contribution = 0
                mean_diet_contribution = 0
                mean_mb_variance = 0
                large_mb_pct = 0
            
            compartment_results.append({
                'compartment': compartment,
                'n_reactions': n,
                'n_significant': k,
                'expected': expected,
                'enrichment_ratio': enrichment_ratio,
                'p_value': p_value,
                'mean_microbiome_contribution': mean_mb_contribution,
                'mean_diet_contribution': mean_diet_contribution,
                'mean_microbiome_variance': mean_mb_variance,
                'large_microbiome_effects_pct': large_mb_pct
            })
        
        # Create DataFrame
        comp_df = pd.DataFrame(compartment_results)
        
        # FDR correction
        comp_df['p_adjusted'] = multipletests(comp_df['p_value'], method='fdr_bh')[1]
        comp_df['significant'] = comp_df['p_adjusted'] < 0.05
        
        # Sort by microbiome contribution
        comp_df = comp_df.sort_values('mean_microbiome_contribution', ascending=False)
        
        # Save results
        output_file = os.path.join(self.output_dir, 'compartment_enrichment_results.csv')
        comp_df.to_csv(output_file, index=False)
        print(f"\n[SAVED] {output_file}")
        
        # Print summary
        print(f"\n[RESULTS] Compartment enrichment:")
        for idx, row in comp_df.iterrows():
            sig_marker = "*" if row['significant'] else " "
            print(f"  {sig_marker} {row['compartment']:15s} {row['n_significant']:3d}/{row['n_reactions']:3d} "
                  f"MB_var={row['mean_microbiome_variance']:5.1f}% "
                  f"(enrichment={row['enrichment_ratio']:.2f}x, p={row['p_adjusted']:.2e})")
        
        self.compartment_enrichment_df = comp_df
        return comp_df
    
    def pathway_synergy_analysis(self):
        """
        Analyze synergy vs antagonism at the pathway level.
        
        Answers: "In which pathways do diet and microbiome cooperate vs compete?"
        
        Returns:
        --------
        pd.DataFrame with pathway-level synergy statistics
        """
        print("\n" + "="*80)
        print("PATHWAY SYNERGY ANALYSIS")
        print("="*80)
        
        synergy_results = []
        
        for subsystem in self.all_subsystems:
            if subsystem == 'Unknown':
                continue
            
            pathway_sig = self.significant_df[
                self.significant_df['subsystem'] == subsystem
            ]
            
            if len(pathway_sig) < 2:
                continue
            
            n = len(pathway_sig)
            n_synergistic = sum(pathway_sig['effect_type'] == 'Synergistic')
            n_antagonistic = sum(pathway_sig['effect_type'] == 'Antagonistic')
            
            synergistic_pct = 100 * n_synergistic / n
            antagonistic_pct = 100 * n_antagonistic / n
            
            # Classify pathway
            if synergistic_pct > 70:
                pathway_class = 'Highly Synergistic'
            elif antagonistic_pct > 70:
                pathway_class = 'Highly Antagonistic'
            elif synergistic_pct > 50:
                pathway_class = 'Moderately Synergistic'
            else:
                pathway_class = 'Mixed'
            
            synergy_results.append({
                'subsystem': subsystem,
                'n_reactions': n,
                'n_synergistic': n_synergistic,
                'n_antagonistic': n_antagonistic,
                'synergistic_pct': synergistic_pct,
                'antagonistic_pct': antagonistic_pct,
                'pathway_class': pathway_class
            })
        
        synergy_df = pd.DataFrame(synergy_results)
        synergy_df = synergy_df.sort_values('synergistic_pct', ascending=False)
        
        # Save results
        output_file = os.path.join(self.output_dir, 'pathway_synergy_analysis.csv')
        synergy_df.to_csv(output_file, index=False)
        print(f"\n[SAVED] {output_file}")
        
        # Print summary
        print(f"\n[RESULTS] Pathway classification:")
        for class_name in ['Highly Synergistic', 'Moderately Synergistic', 'Mixed', 'Highly Antagonistic']:
            count = sum(synergy_df['pathway_class'] == class_name)
            print(f"  {class_name:25s}: {count} pathways")
        
        print(f"\n  Most synergistic pathways:")
        for idx, row in synergy_df.head(5).iterrows():
            print(f"    {row['subsystem'][:50]:50s} {row['synergistic_pct']:5.1f}% synergistic ({row['n_synergistic']}/{row['n_reactions']})")
        
        print(f"\n  Most antagonistic pathways:")
        for idx, row in synergy_df.tail(5).iterrows():
            print(f"    {row['subsystem'][:50]:50s} {row['antagonistic_pct']:5.1f}% antagonistic ({row['n_antagonistic']}/{row['n_reactions']})")
        
        self.synergy_df = synergy_df
        return synergy_df
    
    def visualize_pathway_enrichment(self):
        """
        Create publication-quality pathway enrichment visualizations.
        
        Generates:
        ----------
        1. pathway_enrichment_barplot.png
           - Top 20 enriched pathways
           - Colored by significance
           
        2. compartment_heatmap.png
           - Compartment × metric heatmap
           - Shows microbiome vs diet contributions by compartment
           
        3. synergy_by_pathway.png
           - Stacked bar chart of synergy vs antagonism by pathway
           
        4. pathway_variance_explained.png
           - Scatter plot: diet variance vs microbiome variance by pathway
        """
        print("\n" + "="*80)
        print("GENERATING PATHWAY VISUALIZATIONS")
        print("="*80)
        
        # Figure 1: Pathway Enrichment Bar Plot
        sig_pathways = self.pathway_enrichment_df[
            self.pathway_enrichment_df['significant']
        ].head(20)
        
        if len(sig_pathways) > 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            y_pos = np.arange(len(sig_pathways))
            colors = ['#e74c3c' if r > 2 else '#3498db' for r in sig_pathways['enrichment_ratio']]
            
            ax.barh(y_pos, sig_pathways['enrichment_ratio'], color=colors, edgecolor='black', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([s[:50] for s in sig_pathways['subsystem']], fontsize=9)
            ax.set_xlabel('Enrichment Ratio (Observed/Expected)', fontsize=11, fontweight='bold')
            ax.set_title('Top 20 Enriched Metabolic Pathways', fontsize=13, fontweight='bold')
            ax.axvline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(axis='x', alpha=0.3)
            
            # Add n_significant annotations
            for i, (idx, row) in enumerate(sig_pathways.iterrows()):
                ax.text(row['enrichment_ratio'] + 0.1, i, f"{row['n_significant']}/{row['n_reactions']}", 
                       va='center', fontsize=8)
            
            plt.tight_layout()
            fig_path = os.path.join(self.output_dir, 'pathway_enrichment_barplot.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[SAVED] {fig_path}")
        
        # Figure 2: Compartment Heatmap
        if hasattr(self, 'compartment_enrichment_df'):
            comp_data = self.compartment_enrichment_df.set_index('compartment')[[
                'mean_diet_contribution',
                'mean_microbiome_contribution',
                'mean_microbiome_variance',
                'enrichment_ratio'
            ]]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.heatmap(comp_data.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       cbar_kws={'label': 'Value'}, linewidths=0.5, ax=ax)
            ax.set_ylabel('Metric', fontsize=11, fontweight='bold')
            ax.set_xlabel('Compartment', fontsize=11, fontweight='bold')
            ax.set_title('Compartment Enrichment Profile', fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            fig_path = os.path.join(self.output_dir, 'compartment_heatmap.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[SAVED] {fig_path}")
        
        # Figure 3: Synergy by Pathway
        if hasattr(self, 'synergy_df'):
            top_pathways = self.synergy_df.head(15)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            y_pos = np.arange(len(top_pathways))
            
            ax.barh(y_pos, top_pathways['synergistic_pct'], 
                   label='Synergistic', color='#2ecc71', edgecolor='black', linewidth=0.5)
            ax.barh(y_pos, top_pathways['antagonistic_pct'], 
                   left=top_pathways['synergistic_pct'],
                   label='Antagonistic', color='#e74c3c', edgecolor='black', linewidth=0.5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([s[:50] for s in top_pathways['subsystem']], fontsize=9)
            ax.set_xlabel('Percentage of Reactions (%)', fontsize=11, fontweight='bold')
            ax.set_title('Synergy vs Antagonism by Pathway', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlim(0, 100)
            
            plt.tight_layout()
            fig_path = os.path.join(self.output_dir, 'synergy_by_pathway.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[SAVED] {fig_path}")
        
        # Figure 4: Pathway Variance Explained Scatter
        pathway_variance = self.pathway_enrichment_df[
            self.pathway_enrichment_df['n_significant'] >= 3
        ]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            pathway_variance['mean_diet_variance'],
            pathway_variance['mean_microbiome_variance'],
            s=pathway_variance['n_significant'] * 10,
            c=pathway_variance['synergistic_pct'],
            cmap='RdYlGn',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add diagonal
        ax.plot([0, 100], [100, 0], 'k--', alpha=0.3, linewidth=2, label='Sum = 100%')
        
        # Add quadrant lines
        ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(50, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Mean Diet Variance Explained (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Microbiome Variance Explained (%)', fontsize=11, fontweight='bold')
        ax.set_title('Pathway-Level Variance Partitioning', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('% Synergistic Reactions', fontsize=10)
        
        # Annotate top pathways
        for idx, row in pathway_variance.head(10).iterrows():
            ax.annotate(row['subsystem'][:20], 
                       (row['mean_diet_variance'], row['mean_microbiome_variance']),
                       fontsize=7, alpha=0.7)
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, 'pathway_variance_explained.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {fig_path}")
    
    def generate_publication_summary(self):
        """
        Generate publication-ready text summary.
        
        Creates a comprehensive summary suitable for Results section.
        """
        print("\n" + "="*80)
        print("PUBLICATION SUMMARY")
        print("="*80)
        
        summary = []
        summary.append("\n# PATHWAY-LEVEL ANALYSIS SUMMARY\n")
        summary.append("="*80 + "\n")
        
        # Pathway enrichment
        sig_pathways = self.pathway_enrichment_df[self.pathway_enrichment_df['significant']]
        summary.append(f"\n## Pathway Enrichment\n")
        summary.append(f"- Total pathways analyzed: {len(self.pathway_enrichment_df)}\n")
        summary.append(f"- Significantly enriched: {len(sig_pathways)} (p < 0.05, FDR-corrected)\n")
        
        if len(sig_pathways) > 0:
            summary.append(f"\n### Top 5 Enriched Pathways:\n")
            for idx, row in sig_pathways.head(5).iterrows():
                summary.append(
                    f"  {idx+1}. {row['subsystem']}: {row['n_significant']}/{row['n_reactions']} reactions "
                    f"(enrichment={row['enrichment_ratio']:.2f}x, p={row['p_adjusted']:.2e})\n"
                )
        
        # Compartment enrichment
        if hasattr(self, 'compartment_enrichment_df'):
            sig_comp = self.compartment_enrichment_df[self.compartment_enrichment_df['significant']]
            summary.append(f"\n## Compartment Enrichment\n")
            summary.append(f"- Compartments analyzed: {len(self.compartment_enrichment_df)}\n")
            summary.append(f"- Significantly enriched: {len(sig_comp)}\n")
            
            # Identify compartment with highest microbiome contribution
            top_comp = self.compartment_enrichment_df.iloc[0]
            summary.append(
                f"- Highest microbiome contribution: {top_comp['compartment']} "
                f"({top_comp['mean_microbiome_variance']:.1f}% variance)\n"
            )
        
        # Synergy analysis
        if hasattr(self, 'synergy_df'):
            summary.append(f"\n## Pathway Synergy Classification\n")
            for class_name in ['Highly Synergistic', 'Moderately Synergistic', 'Mixed', 'Highly Antagonistic']:
                count = sum(self.synergy_df['pathway_class'] == class_name)
                if count > 0:
                    summary.append(f"- {class_name}: {count} pathways\n")
            
            # Most synergistic pathway
            most_syn = self.synergy_df.iloc[0]
            summary.append(
                f"\n### Most Synergistic Pathway:\n"
                f"  {most_syn['subsystem']}: {most_syn['synergistic_pct']:.1f}% synergistic "
                f"({most_syn['n_synergistic']}/{most_syn['n_reactions']} reactions)\n"
            )
        
        summary.append("\n" + "="*80 + "\n")
        
        # Save summary
        summary_text = ''.join(summary)
        print(summary_text)
        
        output_file = os.path.join(self.output_dir, 'PATHWAY_ANALYSIS_SUMMARY.txt')
        with open(output_file, 'w') as f:
            f.write(summary_text)
        print(f"\n[SAVED] {output_file}")
        
        return summary_text
    
    def run_complete_analysis(self):
        """
        Run complete pathway enrichment analysis pipeline.
        
        Executes all analysis steps in order:
        1. Pathway enrichment
        2. Compartment enrichment
        3. Synergy analysis
        4. Visualizations
        5. Publication summary
        
        Returns:
        --------
        dict : All results DataFrames
        """
        print("\n" + "="*80)
        print("RUNNING COMPLETE PATHWAY ANALYSIS PIPELINE")
        print("="*80)
        
        results = {}
        
        # Run analyses
        results['pathway_enrichment'] = self.pathway_enrichment_analysis()
        results['compartment_enrichment'] = self.compartment_enrichment_analysis()
        results['pathway_synergy'] = self.pathway_synergy_analysis()
        
        # Generate visualizations
        self.visualize_pathway_enrichment()
        
        # Generate summary
        self.generate_publication_summary()
        
        print("\n" + "="*80)
        print("PATHWAY ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {self.output_dir}")
        
        return results


################################################################################
# COMMAND-LINE INTERFACE
################################################################################

def main():
    """
    Command-line interface for pathway enrichment analysis.
    
    Usage:
    ------
    python rq4_pathway_enrichment_module.py \
        --model iMM1415.json \
        --attribution results_rq4_complete/03_attribution_analysis/flux_attribution_analysis.csv \
        --output results_rq4_complete/03_attribution_analysis/pathway_enrichment
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RQ4 Pathway Enrichment Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python rq4_pathway_enrichment_module.py \\
      --model iMM1415.json \\
      --attribution flux_attribution_analysis.csv \\
      --output pathway_results
  
  # Full RQ4 pipeline integration
  python rq4_pathway_enrichment_module.py \\
      --model iMM1415.json \\
      --attribution results_rq4_complete/03_attribution_analysis/flux_attribution_analysis.csv \\
      --output results_rq4_complete/03_attribution_analysis/pathway_enrichment
        """
    )
    
    parser.add_argument(
        '--model',
        required=True,
        help='Path to iMM1415.json metabolic model file'
    )
    
    parser.add_argument(
        '--attribution',
        required=True,
        help='Path to flux_attribution_analysis.csv from Stage 3'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for pathway enrichment results'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = RQ4PathwayEnrichment(
        model_file=args.model,
        attribution_csv=args.attribution,
        output_dir=args.output
    )
    
    results = analyzer.run_complete_analysis()
    
    #print("\n✓ Analysis complete!")
    #print(f"✓ Results saved to: {args.output}")
    print("\n[Done] Analysis complete!")
    print(f"[Done] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
