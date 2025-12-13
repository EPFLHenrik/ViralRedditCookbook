"""Multi-Treatment Propensity Score Matching Analysis - Excludes VADER."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf
import warnings

# Suppress convergence warnings (we handle them explicitly)
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

# CONFIGURATION

TREATMENT_VARIABLES = [
    'Sentiment_compound_vader',
    'Num_characters',
    'Num_long_words',
    'Num_words',
    'LIWC_Swear',
    'LIWC_They',
    'LIWC_Anger',
    'Avg_word_length',
    'Num_sentences',
    'LIWC_Humans'
]

COVARIATE_EXCLUSIONS = {
    'Sentiment_compound_vader': [
        'Sentiment_pos_vader',
        'Sentiment_neg_vader'
    ],
    'Num_characters': [
        'Num_characters_no_space', 'Num_words', 'Num_sentences',
        'Avg_word_length', 'Avg_chars_per_sentence', 'Avg_words_per_sentence',
        'Num_unique_stopwords', 'Num_long_words', 'Num_unique_words',
        'Num_long_sentences'
    ],
    'Num_characters_no_space': [
        'Num_characters', 'Num_words', 'Num_sentences',
        'Avg_word_length', 'Avg_chars_per_sentence', 'Avg_words_per_sentence',
        'Num_unique_stopwords', 'Num_long_words', 'Num_unique_words',
        'Num_long_sentences'
    ],
    'Num_long_words': [
        'Num_words', 'Avg_word_length', 'Num_characters',
        'Num_characters_no_space', 'Num_unique_words'
    ],
    'Num_words': [
        'Num_characters', 'Num_characters_no_space', 'Num_sentences',
        'Avg_words_per_sentence', 'Num_long_words', 'Avg_word_length'
    ],
    'LIWC_Swear': [],
    'LIWC_They': ['LIWC_Pronoun', 'LIWC_Ppron'],
    'LIWC_Anger': ['LIWC_Negemo', 'LIWC_Affect'],
    'Avg_word_length': [
        'Num_characters', 'Num_characters_no_space', 'Num_words', 'Num_long_words'
    ],
    'Num_sentences': [
        'Num_words', 'Avg_words_per_sentence', 'Avg_chars_per_sentence',
        'Num_characters', 'Num_characters_no_space'
    ],
    'LIWC_Humans': []
}

# Make exclusions symmetric
for treatment_var, exclusions in list(COVARIATE_EXCLUSIONS.items()):
    for excluded_var in exclusions:
        if excluded_var in COVARIATE_EXCLUSIONS:
            if treatment_var not in COVARIATE_EXCLUSIONS[excluded_var]:
                COVARIATE_EXCLUSIONS[excluded_var].append(treatment_var)

TREATMENT_METHOD = 'median'
CUSTOM_THRESHOLDS = {
    'Sentiment_compound_vader': 0  # VADER: positive (>0) = treated, negative (<=0) = control
}
OUTCOME_VAR = 'virality_rss'
CALIPER = 0.00001


# HELPER FUNCTIONS

def prepare_covariates(treatment_var, covariates_list, exclusions):
    """Filter covariates, excluding treatment variable and specified exclusions."""
    return [c for c in covariates_list 
            if c != treatment_var and (exclusions is None or c not in exclusions)]


def fit_propensity_score(df, covariates, maxiter=500, max_attempts=2):
    """Fit logistic regression for propensity scores with retry logic."""
    covariates_to_use = covariates.copy()
    attempt = 1
    res = None
    
    while attempt <= max_attempts:
        formula = 'treat ~ ' + ' + '.join(covariates_to_use)
        mod = smf.logit(formula=formula, data=df)
        res = mod.fit(maxiter=maxiter, method='lbfgs', disp=0)
        
        # Check if converged
        if res.mle_retvals.get('converged', False):
            return res, covariates_to_use
        
        # If not converged and this is the first attempt, try removing problematic covariates
        if attempt == 1:
            print("\n⚠️  First attempt failed to converge. Identifying problematic covariates...")
            coefs = res.params
            nan_coefs = coefs[coefs.isna()].index.tolist()
            inf_coefs = coefs[np.isinf(coefs)].index.tolist()
            large_coefs = coefs[abs(coefs) > 10]
            # Filter out 'Intercept' - it's not a covariate, just the model intercept
            large_coefs_covariates = {k: v for k, v in large_coefs.items() if k != 'Intercept'}
            
            # Combine all problematic covariates (Intercept already excluded from large_coefs_covariates)
            problematic = list(set(nan_coefs + inf_coefs + list(large_coefs_covariates.keys())))
            
            if problematic:
                print(f"  Removing {len(problematic)} problematic covariates: {problematic}")
                covariates_to_use = [c for c in covariates_to_use if c not in problematic]
                print(f"  Retrying with {len(covariates_to_use)} covariates...")
                attempt += 1
                continue
            else:
                print("  No automatically identifiable problematic covariates found.")
                print("  Proceeding to detailed diagnostics...")
                break
        else:
            # Second attempt also failed
            break
    
    # If still not converged after retry, show detailed diagnostics and crash
    if not res.mle_retvals.get('converged', False):
        print("\n" + "="*80)
        print("❌ CONVERGENCE FAILURE - DETAILED DIAGNOSTICS")
        print("="*80)
        
        # Check for NaN/Inf coefficients
        coefs = res.params
        nan_coefs = coefs[coefs.isna()].index.tolist()
        inf_coefs = coefs[np.isinf(coefs)].index.tolist()
        
        print(f"\n1. Convergence status:")
        print(f"   Converged: {res.mle_retvals.get('converged', False)}")
        print(f"   Iterations: {res.mle_retvals.get('iterations', 'N/A')}")
        print(f"   Function evaluations: {res.mle_retvals.get('fcalls', 'N/A')}")
        print(f"   Final function value: {res.mle_retvals.get('fopt', 'N/A')}")
        
        if nan_coefs:
            print(f"\n2. ⚠️  Covariates with NaN coefficients ({len(nan_coefs)}):")
            for cov in nan_coefs:
                print(f"   - {cov}")
                # Check if this covariate has issues
                if cov in df.columns:
                    print(f"     Stats: mean={df[cov].mean():.6f}, std={df[cov].std():.6f}, "
                          f"min={df[cov].min():.6f}, max={df[cov].max():.6f}, "
                          f"nunique={df[cov].nunique()}")
        
        if inf_coefs:
            print(f"\n3. ⚠️  Covariates with Inf coefficients ({len(inf_coefs)}):")
            for cov in inf_coefs:
                print(f"   - {cov}")
                if cov in df.columns:
                    print(f"     Stats: mean={df[cov].mean():.6f}, std={df[cov].std():.6f}, "
                          f"min={df[cov].min():.6f}, max={df[cov].max():.6f}")
        
        # Check for very large coefficients (potential separation)
        # Filter out 'Intercept' - it's not a covariate, just the model intercept
        large_coefs = coefs[abs(coefs) > 10]
        large_coefs_covariates = {k: v for k, v in large_coefs.items() if k != 'Intercept'}
        if len(large_coefs_covariates) > 0:
            print(f"\n4. ⚠️  Covariates with very large coefficients |coef| > 10 ({len(large_coefs_covariates)}):")
            for cov, val in large_coefs_covariates.items():
                print(f"   - {cov}: {val:.4f}")
        # Also show Intercept separately if it's large (for information, but not as a problem)
        if 'Intercept' in large_coefs:
            print(f"\n   (Note: Intercept coefficient is {large_coefs['Intercept']:.4f}, but this is normal and not a covariate)")
        
        # Check condition number of design matrix
        X_matrix = df[covariates_to_use].values
        try:
            from numpy.linalg import cond
            condition_num = cond(X_matrix.T @ X_matrix)
            print(f"\n5. Design matrix condition number: {condition_num:.2e}")
            if condition_num > 1e12:
                print(f"   ⚠️  Very high condition number indicates severe multicollinearity!")
        except:
            print(f"\n5. Could not compute condition number")
        
        # Check for perfect separation
        print(f"\n6. Checking for perfect separation...")
        treated_vals = df[df['treat'] == 1][covariates_to_use]
        control_vals = df[df['treat'] == 0][covariates_to_use]
        for cov in covariates_to_use[:10]:  # Check first 10
            if treated_vals[cov].min() > control_vals[cov].max() or control_vals[cov].min() > treated_vals[cov].max():
                print(f"   ⚠️  Potential separation in {cov}")
        
        print(f"\n7. Suggested problematic covariates to exclude:")
        # Use the filtered large_coefs_covariates (Intercept already excluded)
        problematic = list(set(nan_coefs + inf_coefs + list(large_coefs_covariates.keys())))
        if problematic:
            print(f"   {problematic}")
        else:
            print(f"   (None automatically detected - may need manual investigation)")
        
        print("\n" + "="*80)
        print("❌ STOPPING: Model did not converge after retry with problematic covariates removed.")
        print("="*80)
        
        raise ValueError(f"Logistic regression did not converge even after removing problematic covariates. "
                        f"Problematic covariates: {problematic if problematic else 'See diagnostics above'}")
    
    # Should never reach here, but just in case
    return res, covariates_to_use


def perform_matching(treatment_df, control_df, caliper):
    """Perform nearest-neighbor matching with caliper."""
    treatment_ps = treatment_df['Propensity_score'].values.reshape(-1, 1)
    control_ps = control_df['Propensity_score'].values.reshape(-1, 1)
    
    nn = NearestNeighbors(n_neighbors=1, metric='manhattan')
    nn.fit(control_ps)
    distances, indices = nn.kneighbors(treatment_ps)
    
    valid = distances.flatten() <= caliper
    matched_treatment = treatment_df.index[valid]
    matched_control = control_df.index[indices[valid].flatten()]
    
    return matched_treatment, matched_control


def calculate_effects(treated_matched, control_matched, outcome_var, df_psm):
    """Calculate ATT, top 1% ATT, and balance SMD."""
    att = treated_matched[outcome_var].mean() - control_matched[outcome_var].mean()
    threshold_99 = np.percentile(df_psm[outcome_var], 99)
    treated_top1 = treated_matched[treated_matched[outcome_var] >= threshold_99]
    control_top1 = control_matched[control_matched[outcome_var] >= threshold_99]
    att_top1pct = (treated_top1[outcome_var].mean() - control_top1[outcome_var].mean()
                   if len(treated_top1) > 0 and len(control_top1) > 0 else np.nan)
    ps_t = treated_matched['Propensity_score']
    ps_c = control_matched['Propensity_score']
    pooled_std = np.sqrt((ps_t.std()**2 + ps_c.std()**2) / 2)
    balance_smd = (ps_t.mean() - ps_c.mean()) / pooled_std if pooled_std > 0 else np.nan
    return att, att_top1pct, balance_smd


# MAIN PSM FUNCTION

def run_psm_analysis(treatment_var, df, covariates_list, outcome_var=OUTCOME_VAR,
                     exclusions=None, treatment_method=TREATMENT_METHOD,
                     custom_threshold=None, caliper=CALIPER):
    """
    Run complete PSM analysis for a single treatment variable.
    
    Returns: dict with results or None if analysis fails
    """
    print(f"\n{'='*80}")
    print(f"PSM Analysis: {treatment_var}")
    print(f"{'='*80}")
    
    # Prepare covariates
    covariates_to_use = prepare_covariates(treatment_var, covariates_list, exclusions)
    print(f"Using {len(covariates_to_use)} covariates")
    
    # Create treatment variable
    df_work = df.copy()
    if custom_threshold is not None:
        df_work['treat'] = (df_work[treatment_var] > custom_threshold).astype(int)
    elif treatment_method == 'median':
        threshold = df_work[treatment_var].median()
        df_work['treat'] = (df_work[treatment_var] > threshold).astype(int)
    else:
        raise ValueError(f"Unknown treatment_method: {treatment_method}")
    
    # For VADER specifically: filter out rows where Sentiment_compound_vader == 0
    if treatment_var == 'Sentiment_compound_vader' and custom_threshold == 0:
        n_before = len(df_work)
        df_work = df_work[df_work[treatment_var] != 0].copy()
        if n_before > len(df_work):
            print(f"  Filtered out {n_before - len(df_work):,} rows where {treatment_var} == 0")
    
    # Drop rows with missing values
    df_psm = df_work.dropna(subset=covariates_to_use + ['treat', outcome_var]).copy()
    
    if len(df_psm) == 0:
        print("⚠️  ERROR: No rows after dropping missing values!")
        return None
    
    # Fit propensity score model
    print("Fitting propensity score model...")
    try:
        res, covariates_final = fit_propensity_score(df_psm, covariates_to_use)
        print("✓ Model converged successfully")
    except ValueError as e:
        print(f"❌ {e}")
        return None
    
    df_psm['Propensity_score'] = res.predict()
    
    # Separate groups
    treatment_df = df_psm[df_psm['treat'] == 1].copy()
    control_df = df_psm[df_psm['treat'] == 0].copy()
    
    n_treatment = len(treatment_df)
    n_control = len(control_df)
    print(f"Treatment: {n_treatment:,} | Control: {n_control:,}")
    
    # Matching
    print("Matching...")
    matched_treatment, matched_control = perform_matching(treatment_df, control_df, caliper)
    n_matched = len(matched_treatment)
    print(f"✓ Matched {n_matched:,} pairs")
    
    # Extract matched dataset
    matched_indices = list(matched_treatment) + list(matched_control)
    matched_df = df_psm.loc[matched_indices].copy()
    
    # Calculate treatment effects
    treated_matched = matched_df[matched_df['treat'] == 1]
    control_matched = matched_df[matched_df['treat'] == 0]
    
    att, att_top1pct, balance_smd = calculate_effects(
        treated_matched, control_matched, outcome_var, df_psm
    )
    
    print(f"ATT: {att:.4f} | ATT (top 1%): {att_top1pct:.4f} | Balance SMD: {balance_smd:.4f}")
    
    return {
        'treatment_var': treatment_var,
        'n_treated': n_treatment,
        'n_control': n_control,
        'n_matched': n_matched,
        'att': att,
        'att_top1pct': att_top1pct,
        'balance_smd': balance_smd,
        'matched_df': matched_df
    }


# MULTI-TREATMENT ANALYSIS

def run_multi_treatment_analysis(df, covariates, treatment_variables=None,
                                  covariate_exclusions=None, treatment_method=None,
                                  custom_thresholds=None, outcome_var=None, caliper=None):
    """
    Run PSM analysis for all treatment variables.
    
    Returns: list of result dictionaries
    """
    treatment_variables = treatment_variables or TREATMENT_VARIABLES
    covariate_exclusions = covariate_exclusions or COVARIATE_EXCLUSIONS
    treatment_method = treatment_method or TREATMENT_METHOD
    custom_thresholds = custom_thresholds or CUSTOM_THRESHOLDS
    outcome_var = outcome_var or OUTCOME_VAR
    caliper = caliper or CALIPER
    
    print("="*80)
    print("RUNNING PSM ANALYSIS FOR ALL TREATMENT VARIABLES")
    print("="*80)
    
    all_results = []
    for treatment_var in treatment_variables:
        exclusions = covariate_exclusions.get(treatment_var, [])
        custom_thresh = custom_thresholds.get(treatment_var)
        
        result = run_psm_analysis(
            treatment_var=treatment_var,
            df=df.copy(),
            covariates_list=covariates,
            outcome_var=outcome_var,
            exclusions=exclusions,
            treatment_method=treatment_method,
            custom_threshold=custom_thresh,
            caliper=caliper
        )
        
        if result is not None:
            all_results.append(result)
    
    print(f"\n{'='*80}")
    print(f"✓ Completed {len(all_results)}/{len(treatment_variables)} analyses")
    print(f"{'='*80}")
    
    return all_results


# VISUALIZATION

def visualize_results(results):
    """Create summary visualization of all treatment effects."""
    if len(results) == 0:
        print("⚠️  No results to visualize.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    vars_names = [r['treatment_var'] for r in results]
    
    # 1. ATT comparison
    ax1 = axes[0, 0]
    atts = [r['att'] for r in results]
    colors = ['red' if att < 0 else 'green' for att in atts]
    ax1.barh(vars_names, atts, color=colors, alpha=0.7)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Average Treatment Effect (ATT)')
    ax1.set_title('Treatment Effects: All Variables')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. ATT Top 1% comparison
    ax2 = axes[0, 1]
    atts_top1 = [r['att_top1pct'] for r in results if not np.isnan(r['att_top1pct'])]
    vars_top1 = [r['treatment_var'] for r in results if not np.isnan(r['att_top1pct'])]
    colors_top1 = ['red' if att < 0 else 'green' for att in atts_top1]
    if len(atts_top1) > 0:
        ax2.barh(vars_top1, atts_top1, color=colors_top1, alpha=0.7)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('ATT (Top 1% of Posts)')
        ax2.set_title('Treatment Effects: Top 1% Most Viral Posts')
        ax2.grid(axis='x', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data for top 1%', ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Match rates
    ax3 = axes[1, 0]
    match_rates = [r['n_matched']/r['n_treated']*100 for r in results]
    ax3.barh(vars_names, match_rates, color='steelblue', alpha=0.7)
    ax3.set_xlabel('Match Rate (%)')
    ax3.set_title('Matching Success Rate')
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_xlim(0, 100)
    
    # 4. Balance (SMD)
    ax4 = axes[1, 1]
    balance_smds = [abs(r['balance_smd']) for r in results]
    colors_balance = ['green' if smd < 0.1 else 'orange' if smd < 0.25 else 'red' for smd in balance_smds]
    ax4.barh(vars_names, balance_smds, color=colors_balance, alpha=0.7)
    ax4.axvline(0.1, color='orange', linestyle='--', linewidth=1, label='Acceptable (0.1)')
    ax4.axvline(0.25, color='red', linestyle='--', linewidth=1, label='Poor (0.25)')
    ax4.set_xlabel('|Standardized Mean Difference|')
    ax4.set_title('Covariate Balance After Matching')
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE: Treatment Effects Across All Variables")
    print("="*100)
    print(f"{'Treatment Variable':<20} {'N Treated':<12} {'N Control':<12} {'N Matched':<12} "
          f"{'Match Rate (%)':<15} {'ATT':<12} {'ATT (Top 1%)':<15} {'Balance SMD':<12}")
    print("-"*100)
    
    for r in results:
        match_rate = f"{r['n_matched']/r['n_treated']*100:.1f}"
        att_top1_str = f"{r['att_top1pct']:.4f}" if not np.isnan(r['att_top1pct']) else "N/A"
        print(f"{r['treatment_var']:<20} {r['n_treated']:<12,} {r['n_control']:<12,} "
              f"{r['n_matched']:<12,} {match_rate:<15} {r['att']:<12.4f} "
              f"{att_top1_str:<15} {r['balance_smd']:<12.4f}")
    
    print("="*100)
    print(f"\n✓ Visualization complete: {len(results)} treatment variables analyzed")


# DATA PREPARATION

def prepare_data(df_analysis, outcome_var=OUTCOME_VAR):
    """Prepare df_clean and covariates from df_analysis (matches Cell 17 logic)."""
    exclude = ['treat', 'ups', 'virality_rss', 'score', 'num_comments', 'total_votes',
               'downs', 'subreddit_subscribers', 'Sentiment_pos_vader', 'Sentiment_neg_vader',
               'Sentiment_compound_vader', 'total_awards_received', 'upvote_ratio', 'gilded',
               'num_crossposts', 'is_video', 'created', 'approved_by', 'mod_reason_title',
               'top_awarded_type', 'report_reasons', 'removed_by', 'num_reports',
               'mod_reason_by', 'treatment_tags', 'view_count', 'banned_at_utc', 'likes',
               'banned_by', 'num_duplicates', 'discussion_type', 'mod_note', 'category',
               'approved_at_utc', 'pwls', 'wls', 'Frac_special', 'LIWC_Pronoun',
               'LIWC_Ppron', 'LIWC_Ipron', 'Frac_alpha', 'Frac_digits', 'Frac_whitespace',
               'is_self', 'created_utc', 'Num_unique_words', 'Num_unique_words_no_space']
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    covariates = [c for c in numeric_cols if c not in exclude]
    dropna_subset = covariates + (['treat', outcome_var] if 'treat' in df_analysis.columns else [outcome_var])
    df_clean = df_analysis.dropna(subset=dropna_subset).copy()
    for col in covariates:
        df_clean[col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
    return df_clean, covariates


# MAIN EXECUTION

if __name__ == "__main__":
    import os
    import sys
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    # Load data
    from src.utils.data_utils import load_final_df
    print("Loading data...")
    df_analysis = load_final_df()
    
    # Prepare data
    print("Preparing data...")
    df_clean, covariates = prepare_data(df_analysis)
    print(f"✓ Prepared {len(covariates)} covariates, {len(df_clean):,} rows")
    
    # Run analysis
    results = run_multi_treatment_analysis(df_clean, covariates)
    
    # Visualize
    visualize_results(results)

