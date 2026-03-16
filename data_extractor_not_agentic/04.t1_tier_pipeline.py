#!/usr/bin/env python3
"""
T1: Tier System + Synergy Score Pipeline
=========================================
Data-driven tier classification for polymer-ceramic composite electrolytes.

Key improvements over old tier_analysis_pipeline.py:
- 3-tier baseline hierarchy: same-paper > dataset-wide > literature fallback
- Data-driven threshold calibration (IQR / percentile / sigma methods)
- wt% -> vol% conversion with material densities
- Maxwell-Garnett EMT alongside rule-of-mixtures
- Tracks baseline_source per entry

Usage:
    python t1_tier_pipeline.py --input compiled_with_class.csv --output-dir t1_output/
    python t1_tier_pipeline.py --input compiled_with_class.csv --output-dir t1_output/ --threshold 0.3
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Density lookup (g/cm³) for wt% -> vol% conversion
POLYMER_DENSITIES = {
    "PEO": 1.21, "PVDF": 1.78, "PAN": 1.18, "PPC": 1.3,
    "PMMA": 1.19, "PVC": 1.4, "PEC": 1.3, "PTMC": 1.3, "other": 1.25,
}
CERAMIC_DENSITIES = {
    "LLZO": 5.1, "LATP": 2.9, "LAGP": 2.9, "LLTO": 5.0,
    "SiO2": 2.2, "TiO2": 4.2, "Al2O3": 3.95, "BaTiO3": 6.02,
    "MOF": 1.5, "COF": 1.3, "other": 3.5,
}

# Literature fallback conductivities at RT (S/cm)
LITERATURE_FALLBACK = {
    # polymer+salt baselines
    "PEO/LiTFSI": 1e-6, "PEO/LiClO4": 5e-7, "PEO/LiFSI": 2e-6,
    "PEO/LiPF6": 5e-7, "PEO/LiBF4": 3e-7, "PEO/other": 1e-6, "PEO/none": 1e-7,
    "PVDF/LiTFSI": 1e-7, "PVDF/LiClO4": 5e-8, "PVDF/LiFSI": 2e-7,
    "PVDF/LiPF6": 1e-7, "PVDF/other": 1e-7, "PVDF/none": 1e-8,
    "PAN/LiTFSI": 1e-8, "PAN/LiClO4": 5e-9, "PAN/other": 1e-8, "PAN/none": 1e-9,
    "PPC/LiTFSI": 1e-7, "PPC/other": 1e-7, "PPC/none": 1e-8,
    "PMMA/LiTFSI": 1e-8, "PMMA/LiClO4": 5e-9, "PMMA/other": 1e-8, "PMMA/none": 1e-9,
    "PVC/LiTFSI": 1e-8, "PVC/other": 1e-8, "PVC/none": 1e-9,
    "other/LiTFSI": 1e-7, "other/other": 1e-7, "other/none": 1e-8,
    "PEC/LiTFSI": 1e-7, "PEC/other": 1e-7, "PEC/none": 1e-8,
    "PTMC/LiTFSI": 1e-7, "PTMC/other": 1e-7, "PTMC/none": 1e-8,
    "none/LiTFSI": 1e-7, "none/other": 1e-8, "none/none": 1e-8,
    # ceramic baselines
    "LLZO": 3e-4, "LATP": 1e-4, "LAGP": 1e-4, "LLTO": 5e-5,
    "SiO2": 1e-14, "TiO2": 1e-14, "Al2O3": 1e-14, "BaTiO3": 1e-12,
    "MOF": 1e-8, "COF": 1e-8, "other": 1e-5, "none": 1e-5,
}

# Insulating filler types (σ < 1e-8 S/cm)
INSULATING_FILLERS = {"SiO2", "TiO2", "Al2O3", "BaTiO3"}

TIER_COLORS = {"Tier 1": "#2ecc71", "Tier 2": "#f1c40f", "Tier 3": "#e74c3c"}

# ============================================================================
# Step 1: Data Loading & Cleaning
# ============================================================================

def load_and_filter(input_path: Path) -> pd.DataFrame:
    """Load compiled_with_class.csv and apply RT + conductivity filters."""
    df = pd.read_csv(input_path)
    n_total = len(df)
    logger.info(f"Loaded {n_total} rows from {input_path.name}")

    # Coerce numeric
    df["conductivity_S_cm"] = pd.to_numeric(df["conductivity_S_cm"], errors="coerce")
    df["measurement_temperature_c"] = pd.to_numeric(df["measurement_temperature_c"], errors="coerce")
    df["filler_loading_wt_pct"] = pd.to_numeric(df["filler_loading_wt_pct"], errors="coerce")
    df["filler_loading_vol_pct"] = pd.to_numeric(df["filler_loading_vol_pct"], errors="coerce")

    # Filter conductivity range
    mask_cond = df["conductivity_S_cm"].between(1e-12, 1)
    n_cond_drop = (~mask_cond & df["conductivity_S_cm"].notna()).sum()
    logger.info(f"  Filtered {n_cond_drop} rows outside [1e-12, 1] S/cm")

    # Filter RT
    mask_temp = df["measurement_temperature_c"].between(15, 35)
    n_temp_drop = (~mask_temp & df["measurement_temperature_c"].notna()).sum()
    logger.info(f"  Filtered {n_temp_drop} rows outside [15, 35] °C")

    # Null drops
    mask_null = df["conductivity_S_cm"].notna() & df["measurement_temperature_c"].notna()
    n_null = (~mask_null).sum()
    if n_null > 0:
        logger.info(f"  Filtered {n_null} rows with null conductivity/temperature")

    df = df[mask_cond & mask_temp & mask_null].copy()
    logger.info(f"  Remaining: {len(df)} rows (composites={( df['material_class']=='polymer_ceramic_composite').sum()}, "
                f"pure_polymer={(df['material_class']=='pure_polymer').sum()}, "
                f"pure_ceramic={(df['material_class']=='pure_ceramic').sum()})")
    return df


# ============================================================================
# Step 2: Build Pure-Component Reference Table
# ============================================================================

def _normalize_key(val):
    """Normalize a field value to a clean string key."""
    s = str(val).strip().lower()
    if s in ("", "nan", "none", "na"):
        return "none"
    return s


def build_reference_table(df: pd.DataFrame) -> dict:
    """Build the 3-tier pure component reference table.

    Returns dict with keys: 'polymer', 'ceramic', 'paper_polymer', 'paper_ceramic'
    where paper-level dicts are keyed by (doc_name, polymer_class, salt_type) etc.
    """
    pure_poly = df[df["material_class"] == "pure_polymer"].copy()
    pure_ceram = df[df["material_class"] == "pure_ceramic"].copy()

    # --- Dataset-wide polymer baselines: keyed by (polymer_class, salt_type) ---
    poly_ref = {}
    if not pure_poly.empty:
        for (pc, st), grp in pure_poly.groupby(
            [pure_poly["polymer_class"].apply(_normalize_key),
             pure_poly["salt_type"].apply(_normalize_key)]
        ):
            vals = grp["conductivity_S_cm"].dropna()
            if len(vals) > 0:
                poly_ref[f"{pc}/{st}"] = {
                    "rt_median": float(vals.median()),
                    "rt_mean": float(vals.mean()),
                    "n": int(len(vals)),
                    "q25": float(vals.quantile(0.25)),
                    "q75": float(vals.quantile(0.75)),
                }

    # --- Dataset-wide ceramic baselines: keyed by ceramic_type ---
    ceram_ref = {}
    if not pure_ceram.empty:
        for ct, grp in pure_ceram.groupby(pure_ceram["ceramic_type"].apply(_normalize_key)):
            vals = grp["conductivity_S_cm"].dropna()
            if len(vals) > 0:
                ceram_ref[ct] = {
                    "rt_median": float(vals.median()),
                    "rt_mean": float(vals.mean()),
                    "n": int(len(vals)),
                    "q25": float(vals.quantile(0.25)),
                    "q75": float(vals.quantile(0.75)),
                }

    # --- Same-paper baselines (polymer): keyed by (doc_name, polymer_class, salt_type) ---
    paper_poly = {}
    if not pure_poly.empty:
        for (doc, pc, st), grp in pure_poly.groupby(
            ["doc_name",
             pure_poly["polymer_class"].apply(_normalize_key),
             pure_poly["salt_type"].apply(_normalize_key)]
        ):
            vals = grp["conductivity_S_cm"].dropna()
            if len(vals) > 0:
                paper_poly[(doc, pc, st)] = float(vals.median())

    # --- Same-paper baselines (ceramic): keyed by (doc_name, ceramic_type) ---
    paper_ceram = {}
    if not pure_ceram.empty:
        for (doc, ct), grp in pure_ceram.groupby(
            ["doc_name", pure_ceram["ceramic_type"].apply(_normalize_key)]
        ):
            vals = grp["conductivity_S_cm"].dropna()
            if len(vals) > 0:
                paper_ceram[(doc, ct)] = float(vals.median())

    logger.info(f"  Reference table: {len(poly_ref)} polymer groups, {len(ceram_ref)} ceramic groups")
    logger.info(f"  Same-paper baselines: {len(paper_poly)} polymer, {len(paper_ceram)} ceramic")

    return {
        "polymer": poly_ref,
        "ceramic": ceram_ref,
        "paper_polymer": paper_poly,
        "paper_ceramic": paper_ceram,
    }


def lookup_polymer_baseline(row, ref_table):
    """3-tier hierarchy lookup for polymer baseline conductivity.

    Returns (sigma, source) where source is 'same_paper', 'dataset_wide', or 'literature'.
    """
    doc = row["doc_name"]
    pc = _normalize_key(row.get("polymer_class", "none"))
    st = _normalize_key(row.get("salt_type", "none"))

    # Priority 1: same-paper
    key = (doc, pc, st)
    if key in ref_table["paper_polymer"]:
        return ref_table["paper_polymer"][key], "same_paper"

    # Also try with generic salt
    for salt_try in [st, "none"]:
        key2 = (doc, pc, salt_try)
        if key2 in ref_table["paper_polymer"]:
            return ref_table["paper_polymer"][key2], "same_paper"

    # Priority 2: dataset-wide
    dw_key = f"{pc}/{st}"
    if dw_key in ref_table["polymer"]:
        return ref_table["polymer"][dw_key]["rt_median"], "dataset_wide"
    # Try with generic salt
    dw_key2 = f"{pc}/none"
    if dw_key2 in ref_table["polymer"]:
        return ref_table["polymer"][dw_key2]["rt_median"], "dataset_wide"
    # Try just polymer with any salt
    for k, v in ref_table["polymer"].items():
        if k.startswith(f"{pc}/"):
            return v["rt_median"], "dataset_wide"

    # Priority 3: literature fallback
    lit_key = f"{pc.upper()}/{st}"
    if lit_key in LITERATURE_FALLBACK:
        return LITERATURE_FALLBACK[lit_key], "literature"
    lit_key2 = f"{pc.upper()}/other"
    if lit_key2 in LITERATURE_FALLBACK:
        return LITERATURE_FALLBACK[lit_key2], "literature"

    return None, None


def lookup_ceramic_baseline(row, ref_table):
    """3-tier hierarchy lookup for ceramic baseline conductivity.

    Returns (sigma, source) where source is 'same_paper', 'dataset_wide', or 'literature'.
    """
    doc = row["doc_name"]
    ct = _normalize_key(row.get("ceramic_type", "none"))

    # Priority 1: same-paper
    key = (doc, ct)
    if key in ref_table["paper_ceramic"]:
        return ref_table["paper_ceramic"][key], "same_paper"

    # Priority 2: dataset-wide
    if ct in ref_table["ceramic"]:
        return ref_table["ceramic"][ct]["rt_median"], "dataset_wide"

    # Priority 3: literature fallback
    ct_upper = ct.upper() if ct != "none" else "none"
    if ct_upper in LITERATURE_FALLBACK:
        return LITERATURE_FALLBACK[ct_upper], "literature"
    if ct in LITERATURE_FALLBACK:
        return LITERATURE_FALLBACK[ct], "literature"

    return None, None


# ============================================================================
# Step 3: EMT Calculation
# ============================================================================

def wt_to_vol(wt_frac, rho_polymer, rho_ceramic):
    """Convert weight fraction to volume fraction."""
    if wt_frac is None or np.isnan(wt_frac) or wt_frac <= 0 or wt_frac >= 1:
        return wt_frac
    # φ_vol = (wt/ρ_c) / (wt/ρ_c + (1-wt)/ρ_p)
    vol_filler = wt_frac / rho_ceramic
    vol_polymer = (1 - wt_frac) / rho_polymer
    return vol_filler / (vol_filler + vol_polymer)


def get_filler_vol_fraction(row):
    """Get filler volume fraction, converting from wt% if needed.

    Returns (vol_fraction, fraction_source) where fraction_source is
    'vol_pct', 'wt_pct_converted', or None.
    """
    vol_pct = row.get("filler_loading_vol_pct")
    wt_pct = row.get("filler_loading_wt_pct")

    if pd.notna(vol_pct) and vol_pct > 0:
        return vol_pct / 100.0, "vol_pct"

    if pd.notna(wt_pct) and wt_pct > 0:
        pc = _normalize_key(row.get("polymer_class", "other"))
        ct = _normalize_key(row.get("ceramic_type", "other"))
        rho_p = POLYMER_DENSITIES.get(pc.upper(), POLYMER_DENSITIES.get(pc, 1.25))
        rho_c = CERAMIC_DENSITIES.get(ct.upper(), CERAMIC_DENSITIES.get(ct, 3.5))
        vol_frac = wt_to_vol(wt_pct / 100.0, rho_p, rho_c)
        return vol_frac, "wt_pct_converted"

    return None, None


def calculate_emt_rom(sigma_poly, sigma_ceram, phi):
    """Rule-of-mixtures EMT: σ = (1-φ)σ_poly + φ·σ_ceramic."""
    return (1 - phi) * sigma_poly + phi * sigma_ceram


def calculate_emt_maxwell_garnett(sigma_poly, sigma_ceram, phi):
    """Maxwell-Garnett EMT for dilute spherical inclusions.

    σ_MG = σ_poly * (σ_ceram + 2σ_poly + 2φ(σ_ceram - σ_poly)) /
                     (σ_ceram + 2σ_poly - φ(σ_ceram - σ_poly))
    """
    num = sigma_ceram + 2*sigma_poly + 2*phi*(sigma_ceram - sigma_poly)
    den = sigma_ceram + 2*sigma_poly - phi*(sigma_ceram - sigma_poly)
    if den == 0:
        return None
    return sigma_poly * num / den


def process_composites(df, ref_table):
    """Calculate EMT, synergy scores, and baselines for all composite entries.

    Adds columns: sigma_polymer, sigma_ceramic, sigma_EMT, sigma_MG,
    synergy_score, baseline_source_polymer, baseline_source_ceramic,
    filler_vol_frac, fraction_source, is_insulating_filler.
    """
    composites = df[df["material_class"] == "polymer_ceramic_composite"].copy()
    logger.info(f"Processing {len(composites)} composite entries...")

    # Pre-allocate result columns
    results = {
        "sigma_polymer": [], "sigma_ceramic": [],
        "sigma_EMT": [], "sigma_MG": [],
        "synergy_score": [],
        "baseline_source_polymer": [], "baseline_source_ceramic": [],
        "filler_vol_frac": [], "fraction_source": [],
        "is_insulating_filler": [],
    }

    for idx, row in composites.iterrows():
        sigma_p, src_p = lookup_polymer_baseline(row, ref_table)
        sigma_c, src_c = lookup_ceramic_baseline(row, ref_table)
        phi, frac_src = get_filler_vol_fraction(row)
        ct = _normalize_key(row.get("ceramic_type", "none"))
        is_insulating = ct.upper() in INSULATING_FILLERS

        sigma_exp = row["conductivity_S_cm"]
        sigma_emt = None
        sigma_mg = None
        synergy = None

        if sigma_p is not None and sigma_p > 0:
            if is_insulating:
                # For insulating fillers, baseline is just the polymer
                # σ_ceramic is negligible, EMT ≈ (1-φ)σ_poly
                if phi is not None:
                    sigma_emt = (1 - phi) * sigma_p
                else:
                    sigma_emt = sigma_p
                if sigma_emt > 0 and sigma_exp > 0:
                    synergy = np.log10(sigma_exp / sigma_emt)
            elif sigma_c is not None and sigma_c > 0 and phi is not None:
                sigma_emt = calculate_emt_rom(sigma_p, sigma_c, phi)
                sigma_mg = calculate_emt_maxwell_garnett(sigma_p, sigma_c, phi)
                if sigma_emt > 0 and sigma_exp > 0:
                    synergy = np.log10(sigma_exp / sigma_emt)
            elif phi is None:
                # Missing filler loading — use polymer as baseline reference
                sigma_emt = sigma_p
                if sigma_emt > 0 and sigma_exp > 0:
                    synergy = np.log10(sigma_exp / sigma_emt)

        results["sigma_polymer"].append(sigma_p)
        results["sigma_ceramic"].append(sigma_c)
        results["sigma_EMT"].append(sigma_emt)
        results["sigma_MG"].append(sigma_mg)
        results["synergy_score"].append(synergy)
        results["baseline_source_polymer"].append(src_p)
        results["baseline_source_ceramic"].append(src_c)
        results["filler_vol_frac"].append(phi)
        results["fraction_source"].append(frac_src)
        results["is_insulating_filler"].append(is_insulating)

    for col, vals in results.items():
        composites[col] = vals

    return composites


# ============================================================================
# Step 4: Synergy Score + Tier Assignment (Data-Driven Thresholds)
# ============================================================================

def calibrate_thresholds(synergy_scores, method="auto"):
    """Calibrate tier thresholds from the synergy score distribution.

    Methods:
        'iqr': Tier1 > Q3 + 0.5*IQR, Tier3 < Q1 - 0.5*IQR
        'percentile': Tier1 = top 20%, Tier3 = bottom 20%
        'sigma': Tier1 = mean + 1σ, Tier3 = mean - 1σ
        'auto': pick based on normality test (sigma if Gaussian, else percentile)

    Returns (t1_threshold, t3_threshold, chosen_method, diagnostics).
    """
    scores = synergy_scores.dropna()
    if len(scores) < 10:
        return 0.18, -0.30, "hardcoded", {}

    mean_s = scores.mean()
    std_s = scores.std()
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1

    # Normality test
    if len(scores) >= 20:
        _, p_normal = sp_stats.shapiro(scores.sample(min(500, len(scores)), random_state=42))
    else:
        p_normal = 0

    diagnostics = {
        "mean": float(mean_s), "std": float(std_s),
        "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
        "shapiro_p": float(p_normal), "n": int(len(scores)),
    }

    # Calculate all three
    thresholds = {
        "iqr": (q3 + 0.5 * iqr, q1 - 0.5 * iqr),
        "percentile": (float(scores.quantile(0.80)), float(scores.quantile(0.20))),
        "sigma": (mean_s + std_s, mean_s - std_s),
    }

    diagnostics["all_thresholds"] = {
        k: {"t1": float(v[0]), "t3": float(v[1])} for k, v in thresholds.items()
    }

    if method == "auto":
        # Use sigma if roughly Gaussian (p > 0.01), else percentile
        if p_normal > 0.01:
            method = "sigma"
        else:
            method = "percentile"
        logger.info(f"  Auto-selected method: '{method}' (Shapiro p={p_normal:.4f})")

    t1, t3 = thresholds.get(method, thresholds["percentile"])

    # Validate: if tier split is too extreme, fall back to percentile
    n_t1 = (scores > t1).sum()
    pct_t1 = n_t1 / len(scores) * 100
    if pct_t1 < 5 or pct_t1 > 40:
        logger.warning(f"  {method} gives {pct_t1:.1f}% Tier 1 — falling back to percentile")
        method = "percentile"
        t1, t3 = thresholds["percentile"]

    diagnostics["chosen_method"] = method
    diagnostics["t1_threshold"] = float(t1)
    diagnostics["t3_threshold"] = float(t3)

    return float(t1), float(t3), method, diagnostics


def assign_tiers(composites, t1_thresh, t3_thresh):
    """Assign tiers based on synergy_score and thresholds."""
    tiers = []
    for _, row in composites.iterrows():
        s = row.get("synergy_score")
        phi = row.get("filler_vol_frac")
        src_p = row.get("baseline_source_polymer")

        if s is None or np.isnan(s):
            tiers.append("Unclassified")
        elif src_p is None:
            tiers.append("Unclassified")
        elif phi is None and not row.get("is_insulating_filler", False):
            # Missing filler loading — still classify but flag
            if s > t1_thresh:
                tiers.append("Tier 1*")  # asterisk = uncertain baseline
            elif s < t3_thresh:
                tiers.append("Tier 3*")
            else:
                tiers.append("Tier 2*")
        else:
            if s > t1_thresh:
                tiers.append("Tier 1")
            elif s < t3_thresh:
                tiers.append("Tier 3")
            else:
                tiers.append("Tier 2")

    composites["tier"] = tiers
    return composites


# ============================================================================
# Step 5: Output & Visualization
# ============================================================================

def save_reference_json(ref_table, diagnostics, output_dir):
    """Save pure component reference table as JSON."""
    out = {
        "data_driven": {
            "polymer": ref_table["polymer"],
            "ceramic": ref_table["ceramic"],
        },
        "literature_fallback": LITERATURE_FALLBACK,
        "threshold_diagnostics": diagnostics,
    }
    path = output_dir / "T1_pure_component_reference.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"  Saved reference table to {path}")


def save_tier_summary(composites, diagnostics, output_dir):
    """Save tier distribution summary."""
    tier_counts = composites["tier"].value_counts().to_dict()
    total = len(composites)
    summary = {
        "tier_counts": tier_counts,
        "total_composites": total,
        "tier_percentages": {k: round(v / total * 100, 1) for k, v in tier_counts.items()},
        "threshold_calibration": diagnostics,
        "baseline_source_distribution": {
            "polymer": composites["baseline_source_polymer"].value_counts().to_dict(),
            "ceramic": composites["baseline_source_ceramic"].value_counts().to_dict(),
        },
    }
    path = output_dir / "T1_tier_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Saved tier summary to {path}")
    return summary


def plot_synergy_distribution(composites, t1_thresh, t3_thresh, diagnostics, output_dir):
    """Plot histogram of synergy scores with tier boundaries."""
    scores = composites["synergy_score"].dropna()
    if len(scores) == 0:
        logger.warning("No synergy scores to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(scores, bins=60, color="#3498db", alpha=0.7, edgecolor="white", density=True)

    # Tier boundaries
    ax.axvline(t1_thresh, color=TIER_COLORS["Tier 1"], linewidth=2, linestyle="--",
               label=f"Tier 1 threshold ({t1_thresh:.2f})")
    ax.axvline(t3_thresh, color=TIER_COLORS["Tier 3"], linewidth=2, linestyle="--",
               label=f"Tier 3 threshold ({t3_thresh:.2f})")
    ax.axvline(0, color="black", linewidth=1, linestyle=":", alpha=0.5, label="EMT baseline (0)")

    # Shade regions
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axvspan(t1_thresh, xlim[1], alpha=0.08, color=TIER_COLORS["Tier 1"])
    ax.axvspan(xlim[0], t3_thresh, alpha=0.08, color=TIER_COLORS["Tier 3"])

    # Show all candidate thresholds if available
    all_t = diagnostics.get("all_thresholds", {})
    markers = {"iqr": "v", "percentile": "s", "sigma": "D"}
    chosen = diagnostics.get("chosen_method", "")
    for mname, vals in all_t.items():
        marker = markers.get(mname, "o")
        alpha = 1.0 if mname == chosen else 0.3
        ax.plot(vals["t1"], ylim[1] * 0.95, marker=marker, color=TIER_COLORS["Tier 1"],
                markersize=8, alpha=alpha)
        ax.plot(vals["t3"], ylim[1] * 0.95, marker=marker, color=TIER_COLORS["Tier 3"],
                markersize=8, alpha=alpha, label=f"{mname} thresholds" if mname != chosen else None)

    n_t1 = ((composites["tier"] == "Tier 1") | (composites["tier"] == "Tier 1*")).sum()
    n_t2 = ((composites["tier"] == "Tier 2") | (composites["tier"] == "Tier 2*")).sum()
    n_t3 = ((composites["tier"] == "Tier 3") | (composites["tier"] == "Tier 3*")).sum()
    n_unc = (composites["tier"] == "Unclassified").sum()

    ax.set_xlabel("Synergy Score = log₁₀(σ_measured / σ_EMT)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Synergy Score Distribution (n={len(scores)})\n"
                 f"Tier 1: {n_t1} | Tier 2: {n_t2} | Tier 3: {n_t3} | Unclassified: {n_unc}\n"
                 f"Method: {chosen}",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "T1_tier_distribution.png", dpi=150)
    plt.close()
    logger.info(f"  Saved distribution plot")


def plot_baseline_coverage(composites, output_dir):
    """Bar chart showing baseline source distribution."""
    if len(composites) == 0:
        logger.warning("No composite entries found, skipping baseline coverage plot.")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, title in [
        (axes[0], "baseline_source_polymer", "Polymer Baseline Source"),
        (axes[1], "baseline_source_ceramic", "Ceramic Baseline Source"),
    ]:
        counts = composites[col].fillna("no_match").value_counts()
        colors_map = {
            "same_paper": "#2ecc71", "dataset_wide": "#3498db",
            "literature": "#e67e22", "no_match": "#95a5a6",
        }
        bar_colors = [colors_map.get(c, "#95a5a6") for c in counts.index]
        if len(bar_colors) > 0:
            counts.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white")
        else:
            counts.plot(kind="bar", ax=ax)
            
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

        # Annotate with percentages
        total = counts.sum()
        if total > 0:
            for i, (label, val) in enumerate(counts.items()):
                ax.text(i, val + total * 0.01, f"{val/total*100:.0f}%", ha="center", fontsize=9)

    plt.suptitle("Baseline Coverage for Composite Entries", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "T1_baseline_coverage.png", dpi=150)
    plt.close()
    logger.info(f"  Saved baseline coverage plot")


def plot_emt_scatter(composites, output_dir):
    """Measured vs EMT conductivity scatter colored by tier."""
    valid = composites[
        composites["sigma_EMT"].notna() &
        composites["synergy_score"].notna() &
        composites["tier"].isin(["Tier 1", "Tier 2", "Tier 3"])
    ].copy()

    if len(valid) < 5:
        logger.warning("Not enough data for EMT scatter plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for tier in ["Tier 1", "Tier 2", "Tier 3"]:
        sub = valid[valid["tier"] == tier]
        ax.scatter(sub["sigma_EMT"], sub["conductivity_S_cm"],
                   label=f"{tier} (n={len(sub)})", color=TIER_COLORS[tier],
                   alpha=0.6, edgecolors="white", linewidth=0.5, s=30)

    # 1:1 line
    all_vals = pd.concat([valid["sigma_EMT"], valid["conductivity_S_cm"]])
    lo, hi = all_vals.min() * 0.3, all_vals.max() * 3
    x = np.array([lo, hi])
    ax.plot(x, x, "k--", alpha=0.6, label="1:1 line")
    ax.fill_between(x, x * 0.1, x * 10, alpha=0.04, color="#f1c40f")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("σ_EMT (predicted) [S/cm]", fontsize=11)
    ax.set_ylabel("σ_measured [S/cm]", fontsize=11)
    ax.set_title("Measured vs EMT Conductivity", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.15)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "T1_emt_scatter.png", dpi=150)
    plt.close()
    logger.info(f"  Saved EMT scatter plot")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="T1: Tier System + Synergy Score Pipeline")
    parser.add_argument("--input", type=str, default="actionable_analysis/compiled_with_class.csv",
                        help="Path to compiled_with_class.csv")
    parser.add_argument("--output-dir", type=str, default="actionable_analysis/t1_output",
                        help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Manual symmetric threshold (e.g. 0.3 → Tier1 > +0.3, Tier3 < -0.3)")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "iqr", "percentile", "sigma"],
                        help="Threshold calibration method")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load & Filter ---
    logger.info("=" * 60)
    logger.info("Step 1: Loading & Filtering Data")
    logger.info("=" * 60)
    df = load_and_filter(input_path)
    if df.empty:
        logger.error("No data after filtering. Exiting.")
        return

    # --- Step 2: Build Reference Table ---
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Building Pure-Component Reference Table")
    logger.info("=" * 60)
    ref_table = build_reference_table(df)

    # --- Step 3: EMT Calculation ---
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: EMT Calculation")
    logger.info("=" * 60)
    composites = process_composites(df, ref_table)

    n_with_emt = composites["sigma_EMT"].notna().sum()
    n_with_synergy = composites["synergy_score"].notna().sum()
    logger.info(f"  EMT computed for {n_with_emt}/{len(composites)} entries")
    logger.info(f"  Synergy score computed for {n_with_synergy}/{len(composites)} entries")

    # --- Step 4: Tier Assignment ---
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Synergy Score Analysis & Tier Assignment")
    logger.info("=" * 60)

    if args.threshold is not None:
        t1_thresh = args.threshold
        t3_thresh = -args.threshold
        method = "manual"
        diagnostics = {"chosen_method": "manual", "t1_threshold": t1_thresh, "t3_threshold": t3_thresh}
        logger.info(f"  Using manual thresholds: Tier1 > {t1_thresh}, Tier3 < {t3_thresh}")
    else:
        t1_thresh, t3_thresh, method, diagnostics = calibrate_thresholds(
            composites["synergy_score"], method=args.method
        )
        logger.info(f"  Calibrated thresholds ({method}): Tier1 > {t1_thresh:.3f}, Tier3 < {t3_thresh:.3f}")

    composites = assign_tiers(composites, t1_thresh, t3_thresh)

    # Print tier distribution
    tier_counts = composites["tier"].value_counts()
    total = len(composites)
    logger.info("\n  Tier Distribution:")
    for tier_name in ["Tier 1", "Tier 1*", "Tier 2", "Tier 2*", "Tier 3", "Tier 3*", "Unclassified"]:
        if tier_name in tier_counts.index:
            n = tier_counts[tier_name]
            logger.info(f"    {tier_name}: {n} ({n/total*100:.1f}%)")

    # Baseline source distribution
    logger.info("\n  Baseline Source (polymer):")
    for src, n in composites["baseline_source_polymer"].value_counts().items():
        logger.info(f"    {src}: {n} ({n/total*100:.1f}%)")

    # --- Step 5: Save Outputs ---
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Saving Outputs")
    logger.info("=" * 60)

    # CSV — merge composites back into full dataset
    non_composites = df[df["material_class"] != "polymer_ceramic_composite"].copy()
    # Add missing columns to non_composites so concat works
    for col in composites.columns:
        if col not in non_composites.columns:
            non_composites[col] = np.nan
    full = pd.concat([composites, non_composites], ignore_index=True)
    csv_path = output_dir / "T1_tiered_dataset.csv"
    full.to_csv(csv_path, index=False)
    logger.info(f"  Saved {len(full)} rows to {csv_path}")

    # Reference JSON
    save_reference_json(ref_table, diagnostics, output_dir)

    # Summary JSON
    summary = save_tier_summary(composites, diagnostics, output_dir)

    # Plots
    plot_synergy_distribution(composites, t1_thresh, t3_thresh, diagnostics, output_dir)
    plot_baseline_coverage(composites, output_dir)
    plot_emt_scatter(composites, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
