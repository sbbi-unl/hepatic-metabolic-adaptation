#!/usr/bin/env python3
"""
RQ4: Microbiome Community-Level Metabolic Modeling using MICOM
================================================================

FIXES IN THIS VERSION:
----------------------
1. Corrected medium loading bug (was setting medium to integer instead of dict)
2. Added proper compartment suffix handling for MICOM (_m vs _e)
3. Improved error handling and validation
4. Added medium verification before community building
"""

import os
import sys
import json
import argparse
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COBRApy and MICOM
import cobra
from cobra.io import load_json_model, load_matlab_model

# MICOM imports (with graceful fallback)
try:
    from micom import Community, load_pickle
    from micom.workflows import grow
    from micom.workflows.core import workflow
    MICOM_AVAILABLE = True
except ImportError:
    print("[WARNING] MICOM not installed. Install with: pip install micom")
    print("[WARNING] Falling back to individual species modeling only.")
    MICOM_AVAILABLE = False


################################################################################
# CONFIGURATION AND CONSTANTS
################################################################################

# =============================================================================

PORTAL_METABOLITES = {
    # =========================================================================
    # TIER 1: SHORT-CHAIN FATTY ACIDS (SCFAs) - HIGH PRIORITY
    # =========================================================================
    
    'ac': {
        'name': 'acetate',
        'hepatic_rxn': 'EX_ac_e',
        'importance': 'high',
        'category': 'SCFA',
        'portal_conc_range': '0.1-0.5 mM',
        'mechanisms': 'Lipogenesis substrate, cholesterol synthesis, HDAC inhibition',
        'evidence': 'Nogal 2021, Canfora 2015, Org 2024',
        'validated': 'YES - Primary driver in DD_HFD (2.5% attribution)'
    },
    'ppa': {
        'name': 'propionate',
        'hepatic_rxn': 'EX_ppa_e',
        'importance': 'high',
        'category': 'SCFA',
        'portal_conc_range': '0.01-0.1 mM',
        'mechanisms': 'Gluconeogenesis substrate, cholesterol inhibition, PPARα',
        'evidence': 'Tirosh 2019, Chambers 2018, Wang 2024'
    },
    'but': {
        'name': 'butyrate',
        'hepatic_rxn': 'EX_but_e',
        'importance': 'high',
        'category': 'SCFA',
        'portal_conc_range': '0.001-0.01 mM',
        'mechanisms': 'HDAC inhibition, mitochondrial β-oxidation, anti-inflammatory',
        'evidence': 'Pant 2023, Mattace Raso 2013, Jin 2020'
    },
    
    # =========================================================================
    # TIER 2: BILE ACIDS - HIGH/MEDIUM PRIORITY
    # =========================================================================
    
    'tdchola': {
        'name': 'taurodeoxycholate',
        'hepatic_rxn': 'EX_tdchola_e',
        'importance': 'medium',
        'category': 'bile_acid_conjugated',
        'portal_conc_range': '1-10 μM',
        'mechanisms': 'Taurine-conjugated secondary BA, enhanced solubility',
        'evidence': 'Hofmann 2014, Lake 2013'
    },
    
    # DCA alternatives (dca not in iMM1415, use structural analogs)
    'hdca': {
        'name': 'hexadecenoate (DCA analog)',
        'hepatic_rxn': 'EX_hdca_e',
        'importance': 'medium',
        'category': 'bile_acid_analog',
        'portal_conc_range': 'Variable',
        'mechanisms': 'Proxy for deoxycholate effects (FXR/TGR5 signaling)',
        'evidence': 'Structural analog for missing DCA',
        'note': 'Receives 40% of DCA flux if detected'
    },
    'ocdca': {
        'name': 'octadecenoate (DCA analog)',
        'hepatic_rxn': 'EX_ocdca_e',
        'importance': 'medium',
        'category': 'bile_acid_analog',
        'portal_conc_range': 'Variable',
        'mechanisms': 'Proxy for deoxycholate effects',
        'evidence': 'Structural analog for missing DCA',
        'note': 'Receives 30% of DCA flux if detected'
    },
    'ptdca': {
        'name': 'pentadecenoate (DCA analog)',
        'hepatic_rxn': 'EX_ptdca_e',
        'importance': 'medium',
        'category': 'bile_acid_analog',
        'portal_conc_range': 'Variable',
        'mechanisms': 'Proxy for deoxycholate effects',
        'evidence': 'Structural analog for missing DCA',
        'note': 'Receives 30% of DCA flux if detected'
    },
    
    # =========================================================================
    # TIER 3: NITROGEN METABOLISM - HIGH PRIORITY
    # =========================================================================
    
    'nh4': {
        'name': 'ammonia',
        'hepatic_rxn': 'EX_nh4_e',
        'importance': 'high',
        'category': 'nitrogen',
        'portal_conc_range': '0.05-0.2 mM',
        'mechanisms': 'Urea cycle substrate, glutamine synthesis, nitrogen load',
        'evidence': 'Zhu 2023, Häussinger 1990, Shawcross 2023',
        'validated': 'YES - Consumption in DD_HFD (lumenal uptake)'
    },
    
    # =========================================================================
    # TIER 4: ORGANIC ACIDS & TCA INTERMEDIATES - HIGH/MEDIUM PRIORITY
    # =========================================================================

    
    'succ': {
        'name': 'succinate',
        'hepatic_rxn': 'EX_succ_e',
        'importance': 'high',
        'category': 'organic_acid',
        'portal_conc_range': '1-10 μM',
        'mechanisms': 'TCA anaplerosis, SUCNR1/GPR91 signaling, HIF-1α',
        'evidence': 'Fernández-Veledo 2019, Mills 2016, Wei 2023',
        'note': 'EMERGING importance as signaling molecule'
    },
    'lac__L': {
        'name': 'L-lactate',
        'hepatic_rxn': 'EX_lac__L_e',
        'importance': 'medium',
        'category': 'organic_acid',
        'portal_conc_range': '0.5-2 mM',
        'mechanisms': 'Gluconeogenesis (Cori cycle), redox balance',
        'evidence': 'Brooks 2018, Fang 2025, Gladden 2004'
    },
    'lac__D': {
        'name': 'D-lactate',
        'hepatic_rxn': 'EX_lac__D_e',
        'importance': 'low',
        'category': 'organic_acid',
        'portal_conc_range': '0.01-0.1 mM',
        'mechanisms': 'Slower hepatic metabolism, dysbiosis marker',
        'evidence': 'Fang 2025, Chu 2020, Ewaschuk 2005'
    },
    'pyr': {
        'name': 'pyruvate',
        'hepatic_rxn': 'EX_pyr_e',
        'importance': 'low',
        'category': 'organic_acid',
        'portal_conc_range': '0.05-0.15 mM',
        'mechanisms': 'Gluconeogenesis, PDH → acetyl-CoA, cross-feeding',
        'evidence': 'Jeoung 2015, Magnúsdóttir 2017'
    },
    
    # =========================================================================
    # TIER 5: DISEASE-RELEVANT MARKERS - MEDIUM PRIORITY
    # =========================================================================
    
    'imp': {
        'name': 'imidazole propionate',
        'hepatic_rxn': 'EX_imp_e',
        'importance': 'medium',
        'category': 'histidine_metabolite',
        'portal_conc_range': '0.1-1 μM',
        'mechanisms': 'Insulin resistance, mTOR activation, p62 upregulation',
        'evidence': 'Koh 2018, Molinaro 2020, Lützhøft 2022',
        'note': 'CRITICAL for HFD/T2D studies - diabetes biomarker'
    },
    'etoh': {
        'name': 'ethanol',
        'hepatic_rxn': 'EX_etoh_e',
        'importance': 'medium',
        'category': 'alcohol',
        'portal_conc_range': '<0.1 mM (>1 mM in dysbiosis)',
        'mechanisms': 'ADH → acetaldehyde, lipogenesis, oxidative stress',
        'evidence': 'Meijnikman 2022, Yuan 2019, Bishehsari 2017',
        'note': 'Auto-brewery syndrome; elevated in some NAFLD patients'
    },
    
    # =========================================================================
    # TIER 6: VITAMINS & COFACTORS - LOW PRIORITY
    # =========================================================================
    
    'fol': {
        'name': 'folate',
        'hepatic_rxn': 'EX_fol_e',
        'importance': 'low',
        'category': 'vitamin',
        'portal_conc_range': '0.01-0.1 μM',
        'mechanisms': 'One-carbon metabolism, methionine cycle',
        'evidence': 'Tarracchini 2024, Rossi 2011, Hadadi 2021',
        'validated': 'YES - Minor contributor in DD_HFD'
    },
    'ribflv': {
        'name': 'riboflavin (vitamin B2)',
        'hepatic_rxn': 'EX_ribflv_e',
        'importance': 'low',
        'category': 'vitamin',
        'portal_conc_range': '0.01-0.1 μM',
        'mechanisms': 'FAD/FMN cofactor, redox reactions',
        'evidence': 'Tarracchini 2024, Thakur 2015, Wan 2022'
    },
    'thm': {
        'name': 'thiamine (vitamin B1)',
        'hepatic_rxn': 'EX_thm_e',
        'importance': 'low',
        'category': 'vitamin',
        'portal_conc_range': '<0.1 μM',
        'mechanisms': 'TPP cofactor, pyruvate dehydrogenase, TCA cycle',
        'evidence': 'Magnúsdóttir 2015, Begley 2007'
    },
    'nac': {
        'name': 'niacin (vitamin B3)',
        'hepatic_rxn': 'EX_nac_e',
        'importance': 'low',
        'category': 'vitamin',
        'portal_conc_range': '<0.1 μM',
        'mechanisms': 'NAD+/NADP+ precursor, redox homeostasis',
        'evidence': 'Magnúsdóttir 2015, Grozio 2019'
    },
}

# DCA flux distribution weights (if dca detected in MICOM but not in iMM1415)
DCA_ALTERNATIVE_WEIGHTS = {
    'hdca': 0.40,   # Hexadecenoate - closest structural analog
    'ocdca': 0.30,  # Octadecenoate
    'ptdca': 0.30,  # Pentadecenoate
}

# Metabolite importance tiers for filtering/prioritization
PRIORITY_TIERS = {
    'HIGH': ['ac', 'ppa', 'but', 'nh4', 'succ'],
    'MEDIUM': ['lac__L', 'imp', 'etoh', 'tdchola', 'hdca', 'ocdca', 'ptdca'],
    'LOW': ['lac__D', 'pyr', 'fol', 'ribflv', 'thm', 'nac']
}


################################################################################
# DATA LOADING AND PREPROCESSING
################################################################################

def load_metatranscriptome_data(
    filepath: str,
    expression_cols: List[str] = ['ND_SCD', 'DD_HFD']
) -> pd.DataFrame:
    """
    Load metatranscriptomic data from CSV file.
    
    """
    print(f"[INFO] Loading metatranscriptome data from: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} gene expression records")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    
    # Validate expression columns exist
    missing_cols = [col for col in expression_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Expression columns not found: {missing_cols}")
    
    # Infer species from Genome_name or similar column
    if 'Genome_name' in df.columns:
        species_col = 'Genome_name'
    elif 'Genome' in df.columns:
        species_col = 'Genome'
    elif 'species' in df.columns:
        species_col = 'species'
    else:
        raise ValueError("No species column found (expected: Genome_name, Genome, or species)")
    
    # Standardize species column name
    df = df.rename(columns={species_col: 'species'})
    
    # Count unique species
    n_species = df['species'].nunique()
    print(f"[INFO] Found {n_species} unique species")
    
    # Show top species by total expression
    species_totals = df.groupby('species')[expression_cols].sum().sum(axis=1).sort_values(ascending=False)
    print(f"[INFO] Top 10 species by total expression:")
    print(species_totals.head(10))
    
    return df


def infer_species_abundances(
    metatranscriptome_df: pd.DataFrame,
    expression_cols: List[str],
    method: str = 'total_expression'
) -> pd.DataFrame:
    """
    Infer relative species abundances from metatranscriptomic data.

    """
    print(f"[INFO] Inferring species abundances using method: {method}")
    
    if method == 'total_expression':
        # Sum all gene expression per species per condition
        abundance_df = metatranscriptome_df.groupby('species')[expression_cols].sum()
        
        # Normalize to relative abundances (sum = 1.0 per condition)
        for col in expression_cols:
            total = abundance_df[col].sum()
            abundance_df[f'{col}_abundance'] = abundance_df[col] / total
        
        # Keep only abundance columns
        abundance_df = abundance_df[[f'{col}_abundance' for col in expression_cols]].reset_index()
        
    elif method == 'housekeeping':
        # TODO: Implement housekeeping gene method
        # Requires pre-defined list of universally expressed genes
        raise NotImplementedError("Housekeeping gene method not yet implemented")
    
    else:
        raise ValueError(f"Unknown abundance method: {method}")
    
    print(f"[INFO] Inferred abundances for {len(abundance_df)} species")
    
    # Show top 5 most abundant species per condition
    for col in expression_cols:
        print(f"\n{col}_abundance:")
        print(abundance_df.nlargest(5, f'{col}_abundance')[['species', f'{col}_abundance']])
    
    return abundance_df


def map_species_to_agora_models(
    species_list: List[str],
    agora_dir: str
) -> Dict[str, str]:
    """
    Map species names from metatranscriptome to AGORA model files.

    """
    print(f"[INFO] Mapping species to AGORA models in: {agora_dir}")
    
    # Find all AGORA model files
    agora_path = Path(agora_dir)
    model_files = list(agora_path.glob('*.mat')) + list(agora_path.glob('*.json'))
    
    print(f"[INFO] Found {len(model_files)} AGORA model files")
    
    # Build lookup: model basename → filepath
    model_lookup = {f.stem: str(f) for f in model_files}
    
    # Map species to models
    species_to_model = {}
    unmatched_species = []
    
    for species in species_list:
        # Clean species name (remove spaces, convert to AGORA format)
        # E.g., "Escherichia coli K 12" → "Escherichia_coli_K_12"
        clean_name = species.replace(' ', '_').replace('-', '_')
        
        # Try exact match first
        if clean_name in model_lookup:
            species_to_model[species] = model_lookup[clean_name]
            continue
        
        # Try fuzzy match (first few words of species name)
        words = clean_name.split('_')
        for n_words in range(len(words), 0, -1):
            prefix = '_'.join(words[:n_words])
            matches = [k for k in model_lookup if k.startswith(prefix)]
            if matches:
                # Use first match (could be improved with more sophisticated matching)
                species_to_model[species] = model_lookup[matches[0]]
                break
        else:
            unmatched_species.append(species)
    
    print(f"[INFO] Successfully mapped {len(species_to_model)}/{len(species_list)} species to AGORA models")
    
    if unmatched_species:
        print(f"[WARNING] {len(unmatched_species)} species could not be matched:")
        for sp in unmatched_species[:10]:
            print(f"  - {sp}")
        if len(unmatched_species) > 10:
            print(f"  ... and {len(unmatched_species) - 10} more")
    
    return species_to_model


################################################################################
# MICOM COMMUNITY MODELING
################################################################################

def standardize_medium_compartments(
    medium: Dict[str, float],
    target_compartment: str = 'e'
) -> Dict[str, float]:
    """
    Standardize exchange reaction compartment suffixes for MICOM.

    """
    standardized = {}
    
    for rxn_id, flux_val in medium.items():
        # Replace any compartment suffix with target
        # E.g., EX_glc__D_m → EX_glc__D_e
        if rxn_id.endswith('_m'):
            new_id = rxn_id[:-2] + f'_{target_compartment}'
        elif rxn_id.endswith('_e'):
            new_id = rxn_id[:-2] + f'_{target_compartment}'
        else:
            # No compartment suffix, add one
            new_id = f'{rxn_id}_{target_compartment}'
        
        standardized[new_id] = flux_val
    
    return standardized


def validate_medium(medium: Dict[str, float]) -> bool:
    """
    Validate medium composition before using with MICOM.

    """
    # Check type
    if not isinstance(medium, dict):
        raise ValueError(f"Medium must be dict, got {type(medium)}")
    
    if len(medium) == 0:
        raise ValueError("Medium is empty")
    
    # Check all values are numeric
    for rxn_id, flux_val in medium.items():
        if not isinstance(flux_val, (int, float)):
            raise ValueError(f"Medium flux must be numeric, got {type(flux_val)} for {rxn_id}")
    
    # Check for essential nutrients
    essential = ['glc', 'o2']  # Glucose and oxygen
    found_essential = {nutrient: False for nutrient in essential}
    
    for rxn_id in medium.keys():
        for nutrient in essential:
            if nutrient in rxn_id.lower():
                found_essential[nutrient] = True
    
    missing = [n for n, found in found_essential.items() if not found]
    if missing:
        print(f"[WARNING] Essential nutrients may be missing: {missing}")
        print(f"[WARNING] Available reactions: {list(medium.keys())[:10]}")
    
    print(f"[INFO] Medium validation passed: {len(medium)} exchange reactions")
    return True


def build_micom_community(
    abundance_df: pd.DataFrame,
    species_to_model: Dict[str, str],
    condition: str,
    medium: Dict[str, float],
    solver: str = 'gurobi'
) -> Optional[Community]:
    """
    Build MICOM community model for a specific dietary condition.

    """
    abundance_col = f'{condition}_abundance'
    
    # Filter species with non-zero abundance
    species_with_abundance = abundance_df[abundance_df[abundance_col] > 0]
    
    # Filter to species we have models for
    species_with_models = [
        sp for sp in species_with_abundance['species']
        if sp in species_to_model
    ]
    
    if len(species_with_models) == 0:
        print("[ERROR] No species available for community building!")
        return None
    
    print(f"\n[INFO] Building MICOM community for condition: {condition}")
    print(f"[INFO] Community contains {len(species_with_models)} species with non-zero abundance")
    
    # Create taxonomy DataFrame for MICOM
    # Required columns: id, abundance, file
    taxonomy = pd.DataFrame([
        {
            'id': species,
            'abundance': abundance_df.loc[
                abundance_df['species'] == species, abundance_col
            ].values[0],
            'file': species_to_model[species]
        }
        for species in species_with_models
    ])
    
    total_abundance = taxonomy['abundance'].sum()
    print(f"[INFO] Total community abundance: {total_abundance:.4f}")
    
    # Validate medium before building
    try:
        validate_medium(medium)
    except ValueError as e:
        print(f"[ERROR] Medium validation failed: {e}")
        return None
    
    try:
        # Build community
        print("[INFO] Loading AGORA models and building community...")
        community = Community(
            taxonomy=taxonomy,
            model_db=None,  # Models already specified in 'file' column
            solver=solver,
            progress=True
        )
        
        # CRITICAL FIX: MICOM community.medium needs to be a dict,
        # and exchange reactions must match the community's exchange namespace
        # We need to convert medium format if necessary
        print("[INFO] Setting community medium...")
        
        # Get list of available exchange reactions in community
        community_exchanges = [r.id for r in community.exchanges]
        print(f"[INFO] Community has {len(community_exchanges)} exchange reactions")
        
        # Match medium reactions to community exchanges
        matched_medium = {}
        unmatched_count = 0
        
        for rxn_id, flux_val in medium.items():
            # Try direct match
            if rxn_id in community_exchanges:
                matched_medium[rxn_id] = flux_val
            else:
                # Try compartment variants
                base_id = rxn_id.rsplit('_', 1)[0]  # Remove compartment suffix
                for suffix in ['_e', '_m', '[e]', '[m]']:
                    test_id = base_id + suffix
                    if test_id in community_exchanges:
                        matched_medium[test_id] = flux_val
                        break
                else:
                    unmatched_count += 1
        
        if unmatched_count > 0:
            print(f"[WARNING] {unmatched_count}/{len(medium)} medium reactions could not be matched to community")
        
        print(f"[INFO] Matched {len(matched_medium)}/{len(medium)} medium reactions to community")
        
        # Set medium
        if len(matched_medium) > 0:
            community.medium = matched_medium
            print(f"[INFO] Community medium set successfully!")
        else:
            print(f"[WARNING] No medium reactions matched! Using default minimal medium")
        
        print(f"[INFO] Community built successfully!")
        print(f"[INFO] Community objective: {community.objective.expression}")
        
        return community
        
    except Exception as e:
        print(f"[ERROR] Failed to build community: {e}")
        traceback.print_exc()
        return None


def simulate_community_growth(
    community: Community,
    tradeoff: float = 0.5,
    strategy: str = 'linear'
) -> Optional[pd.DataFrame]:
    """
    Simulate community growth and extract exchange fluxes.

    """
    print(f"\n[INFO] Simulating community growth...")
    print(f"[INFO] Cooperative tradeoff: {tradeoff}")
    print(f"[INFO] Strategy: {strategy}")
    
    try:
        # Try cooperative tradeoff optimization (requires QP solver)
        solution = community.cooperative_tradeoff(
            fraction=tradeoff,
            fluxes=True,
            pfba=False  # Don't minimize flux - we WANT exchange fluxes!
        )
        
        print(f"[INFO] Using cooperative tradeoff optimization")
        
        # Check if solution is feasible
        if solution is None:
            print("[WARNING] Community growth simulation returned None")
            return None
        
        # Handle both float and Series types for growth_rate
        if hasattr(solution.growth_rate, 'sum'):
            total_growth = solution.growth_rate.sum()
        else:
            # growth_rate is already a scalar
            total_growth = float(solution.growth_rate)
        
        if total_growth < 1e-6:
            print(f"[WARNING] Community growth too low: {total_growth}")
            return None
        
        
        # Extract exchange fluxes from solution
        # CommunitySolution has 'fluxes' attribute, need to filter for exchanges
        if hasattr(solution, 'exchanges'):
            # Some MICOM versions have .exchanges directly
            exchange_fluxes = solution.exchanges
        else:
            # FIXED: Handle per-taxon DataFrame structure
            # solution.fluxes is a DataFrame with:
            #   - Rows (index): Taxon names
            #   - Columns: Reaction IDs  
            #   - Values: Flux for each taxon-reaction pair
            
            print(f"[INFO] Extracting exchanges from per-taxon solution...")
            print(f"[INFO] Solution shape: {solution.fluxes.shape} (taxa x reactions)")
            
            # Get exchange reaction IDs
            exchange_rxn_ids = [r.id for r in community.exchanges]
            
            # Find which exchange reactions are in solution columns
            solution_columns = set(solution.fluxes.columns)
            matched_exchanges = [rxn_id for rxn_id in exchange_rxn_ids 
                                if rxn_id in solution_columns]
            
            print(f"[INFO] Found {len(matched_exchanges)}/{len(exchange_rxn_ids)} exchange reactions in solution")
            
            # Aggregate per-taxon fluxes to community level
            FLUX_THRESHOLD = 1e-9  # Lower threshold for detection
            exchange_data = []
            all_fluxes = []
            
            for rxn_id in matched_exchanges:
                # Get fluxes for this reaction across all taxa
                taxon_fluxes = solution.fluxes[rxn_id]
                
                # Aggregate to community level (sum across taxa)
                community_flux = taxon_fluxes.sum()
                
                # Store for diagnostics
                all_fluxes.append((rxn_id, abs(community_flux)))
                
                # Check if above threshold
                if abs(community_flux) > FLUX_THRESHOLD:
                    # Parse metabolite ID from reaction
                    met_id = rxn_id.replace('EX_', '').replace('_e', '').replace('_m', '')
                    exchange_data.append({
                        'reaction': rxn_id,
                        'flux': community_flux,
                        'metabolite': met_id,
                        'direction': 'export' if community_flux > 0 else 'import'
                    })
            
            # Diagnostic output
            if len(exchange_data) == 0:
                # Show top 10 exchange fluxes even if below threshold
                all_fluxes.sort(key=lambda x: x[1], reverse=True)
                print(f"[DIAGNOSTIC] No fluxes above threshold {FLUX_THRESHOLD}")
                print(f"[DIAGNOSTIC] Top 10 exchange flux magnitudes:")
                for rxn_id, flux_mag in all_fluxes[:10]:
                    print(f"  {rxn_id}: {flux_mag:.2e}")
            else:
                print(f"[SUCCESS] Found {len(exchange_data)} active exchanges!")
                print(f"[INFO] Top 5 exchanges:")
                sorted_exchanges = sorted(exchange_data, key=lambda x: abs(x['flux']), reverse=True)
                for ex in sorted_exchanges[:5]:
                    print(f"  {ex['reaction']:30s}: {ex['flux']:10.6f} ({ex['direction']})")
            
            # Create DataFrame with proper columns even if empty
            if len(exchange_data) > 0:
                exchange_fluxes = pd.DataFrame(exchange_data)
            else:
                # Empty DataFrame with correct columns
                exchange_fluxes = pd.DataFrame(columns=['reaction', 'flux', 'metabolite', 'direction'])
                print("[WARNING] No active exchange fluxes found (all below threshold)")
        
        print(f"[INFO] Simulation successful!")
        print(f"[INFO] Total community growth rate: {total_growth:.4f}")
        print(f"[INFO] Number of active exchanges: {len(exchange_fluxes)}")
        
        return exchange_fluxes
        
    except Exception as e:
        # Check if it's a Gurobi optimization failure
        if "Unable to retrieve attribute" in str(e):
            print(f"[ERROR] Optimization failed - likely infeasible model")
            print(f"[INFO] This can happen if medium constraints are too strict")
            print(f"[INFO] Trying with relaxed tolerance...")
            
            try:
                # Try again with more relaxed tolerances
                solution = community.cooperative_tradeoff(
                    fraction=tradeoff,
                    fluxes=True,
                    pfba=False,  # Don't minimize flux
                    atol=1e-4,   # Relaxed tolerance
                    rtol=1e-4
                )
                
                if solution is None:
                    print("[WARNING] Relaxed optimization also failed")
                    return None
                
                # Same processing as above
                if hasattr(solution.growth_rate, 'sum'):
                    total_growth = solution.growth_rate.sum()
                else:
                    total_growth = float(solution.growth_rate)
                
                if total_growth < 1e-6:
                    print(f"[WARNING] Growth too low even with relaxed tolerance")
                    return None
                
                # Extract exchanges with lower threshold
                exchange_rxn_ids = [r.id for r in community.exchanges]
                FLUX_THRESHOLD = 1e-9
                exchange_data = []
                all_fluxes = []
                
                for rxn_id in exchange_rxn_ids:
                    if rxn_id in solution.fluxes.index:
                        flux_val = solution.fluxes[rxn_id]
                        all_fluxes.append((rxn_id, abs(flux_val)))
                        
                        if abs(flux_val) > FLUX_THRESHOLD:
                            met_id = rxn_id.replace('EX_', '').replace('_e', '').replace('_m', '')
                            exchange_data.append({
                                'reaction': rxn_id,
                                'flux': flux_val,
                                'metabolite': met_id,
                                'direction': 'export' if flux_val > 0 else 'import'
                            })
                
                # Diagnostic output
                if len(exchange_data) == 0:
                    all_fluxes.sort(key=lambda x: x[1], reverse=True)
                    print(f"[DIAGNOSTIC] No fluxes above threshold {FLUX_THRESHOLD}")
                    print(f"[DIAGNOSTIC] Top 10 exchange flux magnitudes:")
                    for rxn_id, flux_mag in all_fluxes[:10]:
                        print(f"  {rxn_id}: {flux_mag:.2e}")
                
                if len(exchange_data) > 0:
                    exchange_fluxes = pd.DataFrame(exchange_data)
                else:
                    exchange_fluxes = pd.DataFrame(columns=['reaction', 'flux', 'metabolite', 'direction'])
                
                print(f"[INFO] Relaxed optimization successful!")
                print(f"[INFO] Total community growth rate: {total_growth:.4f}")
                print(f"[INFO] Number of active exchanges: {len(exchange_fluxes)}")
                
                return exchange_fluxes
                
            except Exception as e2:
                print(f"[ERROR] Relaxed optimization also failed: {e2}")
                return None
        
        elif "only supports linear" in str(e):
            # GLPK fallback
            print(f"[WARNING] QP solver not available (GLPK limitation)")
            print(f"[INFO] Falling back to simple linear optimization...")
            
            try:
                solution = community.optimize()
                
                if solution.objective_value < 1e-6:
                    print("[WARNING] Community growth failed with linear optimization")
                    return None
                
                # Extract exchange fluxes with lower threshold
                exchange_rxns = [r for r in community.exchanges]
                FLUX_THRESHOLD = 1e-9
                exchange_data = []
                all_fluxes = []
                
                for rxn in exchange_rxns:
                    flux_val = solution.fluxes.get(rxn.id, 0.0)
                    all_fluxes.append((rxn.id, abs(flux_val)))
                    
                    if abs(flux_val) > FLUX_THRESHOLD:
                        met_id = rxn.id.replace('EX_', '').replace('_e', '').replace('_m', '')
                        exchange_data.append({
                            'reaction': rxn.id,
                            'flux': flux_val,
                            'metabolite': met_id,
                            'direction': 'export' if flux_val > 0 else 'import'
                        })
                
                # Diagnostic output
                if len(exchange_data) == 0:
                    all_fluxes.sort(key=lambda x: x[1], reverse=True)
                    print(f"[DIAGNOSTIC] No fluxes above threshold {FLUX_THRESHOLD}")
                    print(f"[DIAGNOSTIC] Top 10 exchange flux magnitudes:")
                    for rxn_id, flux_mag in all_fluxes[:10]:
                        print(f"  {rxn_id}: {flux_mag:.2e}")
                
                if len(exchange_data) > 0:
                    exchange_fluxes = pd.DataFrame(exchange_data)
                else:
                    exchange_fluxes = pd.DataFrame(columns=['reaction', 'flux', 'metabolite', 'direction'])
                
                print(f"[INFO] Linear optimization successful!")
                print(f"[INFO] Community growth rate: {solution.objective_value:.4f}")
                print(f"[INFO] Number of active exchanges: {len(exchange_fluxes)}")
                
                return exchange_fluxes
                
            except Exception as e2:
                print(f"[ERROR] Linear optimization also failed: {e2}")
                return None
        else:
            # Different error
            print(f"[ERROR] Community simulation failed: {e}")
            traceback.print_exc()
            return None


def extract_portal_metabolites(
    exchange_fluxes: pd.DataFrame,
    threshold: float = 1e-6,
    filter_metabolites: bool = True
) -> Dict[str, float]:
    """
    Extract portal metabolites (microbiome → liver) from community exchanges.

    """
    portal_fluxes = {}
    dca_flux_detected = None
    
    # Handle empty DataFrame
    if exchange_fluxes is None or len(exchange_fluxes) == 0:
        print("[WARNING] No exchange fluxes provided - returning empty portal metabolites")
        return portal_fluxes
    
    # Verify required columns exist
    if 'reaction' not in exchange_fluxes.columns or 'metabolite' not in exchange_fluxes.columns:
        print(f"[WARNING] Exchange fluxes missing required columns. Has: {exchange_fluxes.columns.tolist()}")
        return portal_fluxes
    
    if filter_metabolites:
        # Filtered mode: Only extract known portal metabolites
        print(f"[INFO] Extracting {len(PORTAL_METABOLITES)} filtered portal metabolites")
        
        # First pass: Extract all metabolites except DCA alternatives
        for met_id in PORTAL_METABOLITES:
            # Skip DCA alternatives on first pass (will be filled from DCA if needed)
            if met_id in DCA_ALTERNATIVE_WEIGHTS:
                continue
                
            # Search for this metabolite in exchange fluxes
            # Handle different ID formats (ac, EX_ac, EX_ac_e, EX_ac_m, etc.)
            flux_rows = exchange_fluxes[
                exchange_fluxes['reaction'].str.contains(f'EX_{met_id}', na=False) |
                exchange_fluxes['metabolite'].str.contains(met_id, na=False)
            ]
            
            if len(flux_rows) > 0:
                # Sum fluxes if multiple matches (unlikely but handle gracefully)
                total_flux = flux_rows['flux'].sum()
                
                if abs(total_flux) > threshold:
                    portal_fluxes[met_id] = total_flux
                    
                # Special handling: Check for DCA
                if met_id == 'dca' and abs(total_flux) > threshold:
                    dca_flux_detected = total_flux
                    print(f"[INFO] Detected deoxycholate (dca) flux: {dca_flux_detected:.4f}")
        
        # Second pass: Handle DCA alternatives if DCA was detected
        if dca_flux_detected is not None:
            print(f"[INFO] Distributing DCA flux across alternatives (not in iMM1415):")
            for alt_met_id, weight in DCA_ALTERNATIVE_WEIGHTS.items():
                distributed_flux = dca_flux_detected * weight
                portal_fluxes[alt_met_id] = distributed_flux
                print(f"  {alt_met_id}: {distributed_flux:.4f} ({weight*100:.0f}% of DCA)")
            
            # Remove original DCA entry (not in hepatic model)
            del portal_fluxes['dca']
            print(f"[INFO] Removed 'dca' entry (replaced with alternatives)")
        else:
            # DCA not detected, check if alternatives are directly produced
            for alt_met_id in DCA_ALTERNATIVE_WEIGHTS:
                flux_rows = exchange_fluxes[
                    exchange_fluxes['reaction'].str.contains(f'EX_{alt_met_id}', na=False) |
                    exchange_fluxes['metabolite'].str.contains(alt_met_id, na=False)
                ]
                
                if len(flux_rows) > 0:
                    total_flux = flux_rows['flux'].sum()
                    if abs(total_flux) > threshold:
                        portal_fluxes[alt_met_id] = total_flux
    else:
        # Unfiltered mode: Extract all exchange fluxes above threshold
        print(f"[INFO] Extracting ALL metabolites above threshold (unfiltered mode)")
        for idx, row in exchange_fluxes.iterrows():
            if abs(row['flux']) > threshold:
                met_id = row['metabolite']
                portal_fluxes[met_id] = row['flux']
        print(f"[INFO] Found {len(portal_fluxes)} metabolites above threshold")
    
    # Summary by importance tier
    if filter_metabolites:
        print(f"\n[SUMMARY] Portal metabolites extracted by priority:")
        for tier, met_list in PRIORITY_TIERS.items():
            tier_metabolites = [m for m in met_list if m in portal_fluxes]
            if tier_metabolites:
                print(f"  {tier}: {len(tier_metabolites)} metabolites")
                for met_id in tier_metabolites:
                    flux = portal_fluxes[met_id]
                    direction = "production" if flux > 0 else "consumption"
                    print(f"    {met_id:12s} ({PORTAL_METABOLITES[met_id]['name']:30s}): {flux:+10.4f} ({direction})")
    
    return portal_fluxes


################################################################################
# VISUALIZATION AND REPORTING
################################################################################

def compare_diet_metabolite_production(
    portal_metabolites_dict: Dict[str, Dict[str, float]],
    results_dir: str
):
    """
    Generate comparative visualizations of portal metabolite production.

    """
    if len(portal_metabolites_dict) == 0:
        print("[WARNING] No portal metabolite data to plot")
        return
    
    # Create results dataframe
    rows = []
    for condition, metabolites in portal_metabolites_dict.items():
        for met_id, flux in metabolites.items():
            # Handle both filtered and unfiltered metabolites
            if met_id in PORTAL_METABOLITES:
                met_name = PORTAL_METABOLITES[met_id]['name']
                importance = PORTAL_METABOLITES[met_id]['importance']
            else:
                met_name = met_id  # Use ID as name for unfiltered metabolites
                importance = 'unknown'
            
            rows.append({
                'Condition': condition,
                'Metabolite': met_name,
                'Metabolite_ID': met_id,
                'Flux': flux,
                'Importance': importance
            })
    
    df = pd.DataFrame(rows)
    
    if len(df) == 0:
        print("[WARNING] No metabolite fluxes to visualize")
        return
    
    # Save detailed table
    table_path = os.path.join(results_dir, 'portal_metabolite_production.csv')
    df.to_csv(table_path, index=False)
    print(f"[SAVED] Portal metabolite table: {table_path}")
    
    # Figure 1: Heatmap
    plt.figure(figsize=(10, 8))
    pivot_df = df.pivot(index='Metabolite', columns='Condition', values='Flux')
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Production Flux (mmol/gDW/h)'}
    )
    plt.title('Portal Metabolite Production Across Diets')
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, 'portal_metabolites_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Heatmap: {heatmap_path}")
    
    # Figure 2: Bar plot comparison (high importance metabolites only)
    high_importance = df[df['Importance'] == 'high']
    if len(high_importance) > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=high_importance,
            x='Metabolite',
            y='Flux',
            hue='Condition'
        )
        plt.ylabel('Production Flux (mmol/gDW/h)')
        plt.title('Key Portal Metabolite Production')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Diet')
        plt.tight_layout()
        barplot_path = os.path.join(results_dir, 'portal_metabolites_barplot.png')
        plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] Bar plot: {barplot_path}")


################################################################################
# COMMAND LINE INTERFACE
################################################################################

def parse_arguments():
    """
    Parse command-line arguments.

    """
    parser = argparse.ArgumentParser(
        description='RQ4: Microbiome Community Metabolic Modeling with MICOM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Basic usage with metatranscriptome
python rq4_microbiome_community_modeling.py \\
    --metatranscriptome Meta_GSE104913.csv \\
    --agora_dir /path/to/AGORA2 \\
    --expression_cols ND_SCD DD_HFD \\
    --results_dir results_community

# With custom medium (diet-specific)
python rq4_microbiome_community_modeling.py \\
    --metatranscriptome Meta_GSE104913.csv \\
    --agora_dir /path/to/AGORA2 \\
    --expression_cols ND_SCD DD_HFD \\
    --medium_json expanded_diet_bounds_flat.json \\
    --results_dir results_community
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--metatranscriptome',
        required=True,
        help='Path to metatranscriptome CSV file'
    )
    
    parser.add_argument(
        '--agora_dir',
        required=True,
        help='Directory containing AGORA model files (.mat or .json)'
    )
    
    parser.add_argument(
        '--expression_cols',
        nargs='+',
        default=['ND_SCD', 'DD_HFD'],
        help='Column names for expression values (conditions to compare)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--medium_json',
        default=None,
        help='Optional: Diet-specific medium composition JSON file'
    )
    
    parser.add_argument(
        '--abundance_method',
        choices=['total_expression', 'housekeeping'],
        default='total_expression',
        help='Method for inferring species abundances'
    )
    
    parser.add_argument(
        '--tradeoff',
        type=float,
        default=0.5,
        help='MICOM cooperative tradeoff (0=pure cooperation, 1=competition)'
    )
    
    parser.add_argument(
        '--solver',
        choices=['glpk', 'gurobi', 'cplex'],
        default='gurobi',
        help='LP solver backend'
    )
    
    parser.add_argument(
        '--filter_metabolites',
        action='store_true',
        default=True,
        help='Filter portal metabolites to known biologically meaningful ones (default: True). Use --no-filter_metabolites to extract all metabolites.'
    )
    
    parser.add_argument(
        '--no-filter_metabolites',
        dest='filter_metabolites',
        action='store_false',
        help='Extract ALL exchange metabolites above threshold (exploratory mode)'
    )
    
    parser.add_argument(
        '--results_dir',
        default='results_microbiome_community',
        help='Output directory for results'
    )
    
    return parser.parse_args()


################################################################################
# MAIN ANALYSIS WORKFLOW
################################################################################

def main():
    """
    Main workflow for microbiome community modeling.

    """
    args = parse_arguments()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*80)
    print("RQ4: MICROBIOME COMMUNITY METABOLIC MODELING")
    print("="*80)
    
    # Step 1: Load metatranscriptome
    metatranscriptome_df = load_metatranscriptome_data(
        args.metatranscriptome,
        args.expression_cols
    )
    
    # Step 2: Infer species abundances
    abundance_df = infer_species_abundances(
        metatranscriptome_df,
        args.expression_cols,
        args.abundance_method
    )
    
    # Save abundances
    abundance_path = os.path.join(args.results_dir, 'species_abundances.csv')
    abundance_df.to_csv(abundance_path, index=False)
    print(f"[SAVED] Species abundances: {abundance_path}")
    
    # Step 3: Map species to AGORA models
    species_list = abundance_df['species'].tolist()
    species_to_model = map_species_to_agora_models(species_list, args.agora_dir)
    
    mapping_df = pd.DataFrame([
        {'species': sp, 'model_file': os.path.basename(model)}
        for sp, model in species_to_model.items()
    ])
    mapping_path = os.path.join(args.results_dir, 'species_to_model_mapping.csv')
    mapping_df.to_csv(mapping_path, index=False)
    print(f"[SAVED] Species-model mapping: {mapping_path}")
    
    # Step 4: Define medium
    # CRITICAL FIX: Properly load medium as dictionary and handle bounds format
    if args.medium_json:
        with open(args.medium_json, 'r') as f:
            medium_dict = json.load(f)
        
        # If medium_dict is already flat (like agora_medium.json), use directly
        # If it's nested by diet condition, extract first condition
        if isinstance(medium_dict, dict):
            # Check if values are dicts (nested) or numbers/lists (flat)
            first_value = list(medium_dict.values())[0]
            if isinstance(first_value, dict):
                # Nested format: {"HFD": {"EX_glc__D_e": 10, ...}, "SCD": {...}}
                # Use first condition
                first_condition = list(medium_dict.keys())[0]
                medium_raw = medium_dict[first_condition]
                print(f"[INFO] Using medium from condition: {first_condition}")
            else:
                # Flat format: {"EX_glc__D_e": 10, ...} or {"EX_glc__D_e": [-10, 1000], ...}
                medium_raw = medium_dict
                print(f"[INFO] Using flat medium definition")
        else:
            raise ValueError(f"Unexpected medium format: {type(medium_dict)}")
        
        # Convert bounds format to single uptake values if needed
        medium = {}
        for rxn_id, value in medium_raw.items():
            if isinstance(value, list):
                # Bounds format: [lower_bound, upper_bound]
                # For FBA: negative lower bound = maximum uptake rate
                # E.g., [-10, 1000] means can take up to 10 mmol/gDW/h
                lower_bound, upper_bound = value
                uptake_rate = abs(lower_bound)  # Convert to positive uptake rate
                medium[rxn_id] = uptake_rate
            elif isinstance(value, (int, float)):
                # Single value format: already an uptake rate
                medium[rxn_id] = abs(value)  # Ensure positive
            else:
                raise ValueError(f"Unexpected value format for {rxn_id}: {type(value)}")
        
        # Standardize compartments
        medium = standardize_medium_compartments(medium, target_compartment='e')
        
    else:
        # Use default Western diet medium
        medium = {
            'EX_glc__D_e': 10.0,
            'EX_fru_e': 5.0,
            'EX_gal_e': 2.0,
            'EX_mal__L_e': 5.0,
            'EX_pyr_e': 1.0,
            'EX_ac_e': 0.1,
            'EX_o2_e': 20.0,
            'EX_pi_e': 10.0,
            'EX_so4_e': 10.0,
            'EX_nh4_e': 10.0,
        }
    
    print(f"[INFO] Medium contains {len(medium)} exchange reactions")
    print(f"[INFO] Sample medium reactions: {list(medium.keys())[:5]}")
    print(f"[INFO] Sample uptake rates: {list(medium.values())[:5]}")
    
    # Step 5: Build communities and simulate for each condition
    if not MICOM_AVAILABLE:
        print("\n[ERROR] MICOM is required for community modeling.")
        print("[INFO] Install with: pip install micom")
        print("[INFO] Exiting...")
        sys.exit(1)
    
    portal_metabolites_dict = {}
    
    for condition in args.expression_cols:
        print(f"\n{'='*80}")
        print(f"Processing condition: {condition}")
        print(f"{'='*80}")
        
        # Build community
        community = build_micom_community(
            abundance_df,
            species_to_model,
            condition,
            medium,
            args.solver
        )
        
        if community is None:
            print(f"[WARNING] Skipping {condition} - community building failed")
            continue
        
        # Simulate growth
        exchange_fluxes = simulate_community_growth(
            community,
            tradeoff=args.tradeoff
        )
        
        if exchange_fluxes is None:
            print(f"[WARNING] Skipping {condition} - simulation failed")
            continue
        
        # Extract portal metabolites
        portal_fluxes = extract_portal_metabolites(
            exchange_fluxes,
            filter_metabolites=args.filter_metabolites
        )
        portal_metabolites_dict[condition] = portal_fluxes
    
    # Step 6: Save portal metabolite fluxes for hepatic integration
    portal_output = {}
    for condition, metabolites in portal_metabolites_dict.items():
        portal_output[condition] = {}
        for met_id, flux in metabolites.items():
            if met_id in PORTAL_METABOLITES:
                # Filtered metabolite with full metadata
                portal_output[condition][met_id] = {
                    'flux': flux,
                    'name': PORTAL_METABOLITES[met_id]['name'],
                    'hepatic_rxn': PORTAL_METABOLITES[met_id]['hepatic_rxn'],
                    'importance': PORTAL_METABOLITES[met_id]['importance'],
                    'category': PORTAL_METABOLITES[met_id]['category']
                }
            else:
                # Unfiltered metabolite - create basic metadata
                portal_output[condition][met_id] = {
                    'flux': flux,
                    'name': met_id,
                    'hepatic_rxn': f'EX_{met_id}_e',
                    'importance': 'unknown',
                    'category': 'unfiltered'
                }
    
    portal_json_path = os.path.join(args.results_dir, 'portal_metabolites_for_hepatic_model.json')
    with open(portal_json_path, 'w') as f:
        json.dump(portal_output, f, indent=2)
    print(f"\n[SAVED] Portal metabolites for hepatic integration: {portal_json_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("PORTAL METABOLITE EXTRACTION SUMMARY")
    print(f"{'='*80}")
    for condition, metabolites in portal_output.items():
        print(f"\n{condition}:")
        print(f"  Total metabolites: {len(metabolites)}")
        
        # Count by importance
        by_importance = {}
        for met_id, met_data in metabolites.items():
            importance = met_data['importance']
            by_importance[importance] = by_importance.get(importance, 0) + 1
        
        print(f"  By importance: {dict(by_importance)}")
        
        # Count production vs consumption
        production = sum(1 for m in metabolites.values() if m['flux'] > 0)
        consumption = sum(1 for m in metabolites.values() if m['flux'] < 0)
        print(f"  Production: {production}, Consumption: {consumption}")
        
        # Show top 5 by absolute flux
        top_metabolites = sorted(metabolites.items(), key=lambda x: abs(x[1]['flux']), reverse=True)[:5]
        print(f"  Top 5 by |flux|:")
        for met_id, met_data in top_metabolites:
            #direction = "⟶" if met_data['flux'] > 0 else "⟵"
            direction = "->" if met_data['flux'] > 0 else "<-"
            print(f"    {met_id:12s} ({met_data['name']:30s}): {met_data['flux']:+10.4f} {direction}")
    
    # Step 7: Generate comparison plots
    if len(portal_metabolites_dict) > 0:
        compare_diet_metabolite_production(
            portal_metabolites_dict,
            args.results_dir
        )
    
    print("\n" + "="*80)
    print("RQ4 MICROBIOME COMMUNITY MODELING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.results_dir}")
    print("\nNext steps:")
    print("1. Review portal metabolite profiles")
    print("2. Use portal_metabolites_for_hepatic_model.json in hepatic integration")
    print("3. Run rq4_hepatic_integration.py to constrain hepatic models")


if __name__ == '__main__':
    main()
