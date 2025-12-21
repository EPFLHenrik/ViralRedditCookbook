"""
PSM analysis functions.
"""

import os
import gzip
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

OUTCOME_VAR = 'virality_rss'
CALIPER = 0.00001
EPS = 1e-6
SUPPORT_Q = 0.01
CALIPER_MULT = 0.05

COVARIATE_EXCLUSIONS = {
    'Num_characters': [
        'Num_characters_no_space', 'Num_words', 'Num_sentences',
        'Avg_word_length', 'Avg_chars_per_sentence', 'Avg_words_per_sentence',
        'Num_unique_stopwords', 'Num_long_words', 'Num_unique_words',
        'Num_long_sentences'
    ]
}


def filter_covariates(treatment_var, covariates_list, exclusions):
    return [c for c in covariates_list
            if c != treatment_var and (exclusions is None or c not in exclusions)]


def calculate_smd(x_treated, x_control):
    x_treated = np.asarray(x_treated, dtype=float)
    x_control = np.asarray(x_control, dtype=float)
    pooled_std = np.sqrt((np.nanstd(x_treated, ddof=1)**2 + np.nanstd(x_control, ddof=1)**2) / 2)
    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan
    return (np.nanmean(x_treated) - np.nanmean(x_control)) / pooled_std


def fit_propensity_model(df, covariates):
    formula = 'treat ~ ' + ' + '.join(covariates)
    mod = smf.logit(formula=formula, data=df)
    res = mod.fit(maxiter=500, method='lbfgs', disp=0)
    return res


def compute_propensity_scores(df, model, eps=EPS):
    ps = np.clip(model.predict(), eps, 1 - eps)
    df['Propensity_score'] = ps
    df['logit_ps'] = np.log(ps / (1 - ps))
    return df


def create_treatment_indicator(df, treatment_var):
    threshold = df[treatment_var].median()
    df['treat'] = (df[treatment_var] > threshold).astype(int)
    return df, threshold


def calculate_common_support_bounds(df_psm, support_q=SUPPORT_Q):
    t_ps = df_psm.loc[df_psm['treat'] == 1, 'Propensity_score']
    c_ps = df_psm.loc[df_psm['treat'] == 0, 'Propensity_score']
    
    if len(t_ps) == 0 or len(c_ps) == 0:
        return None, None
    
    t_lo, t_hi = t_ps.quantile(support_q), t_ps.quantile(1 - support_q)
    c_lo, c_hi = c_ps.quantile(support_q), c_ps.quantile(1 - support_q)
    support_low = max(t_lo, c_lo)
    support_high = min(t_hi, c_hi)
    return support_low, support_high


def trim_to_common_support(df_psm, support_low, support_high):
    return df_psm[(df_psm['Propensity_score'] >= support_low) & 
                  (df_psm['Propensity_score'] <= support_high)].copy()


def calculate_caliper(df_psm, caliper_mult=CALIPER_MULT, caliper_fallback=CALIPER):
    logit_sd = df_psm['logit_ps'].std()
    caliper = (caliper_mult * logit_sd) if np.isfinite(logit_sd) and logit_sd > 0 else caliper_fallback
    return caliper


def perform_nearest_neighbor_matching(treatment_df, control_df, caliper, score_col="logit_ps"):
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
    nn.fit(control_df[score_col].values.reshape(-1, 1))
    distances, indices = nn.kneighbors(treatment_df[score_col].values.reshape(-1, 1))
    
    distances = distances.flatten()
    indices = indices.flatten()
    valid = distances <= caliper
    
    matched_t_idx = treatment_df.index[valid]
    matched_c_idx = control_df.index[indices[valid]]
    return matched_t_idx, matched_c_idx


def calculate_covariate_balance(treatment_df, control_df, matched_t_idx, matched_c_idx, covariates, df_psm):
    cov_smds = []
    for cov in covariates:
        smd_before = calculate_smd(treatment_df[cov], control_df[cov])
        smd_after = calculate_smd(df_psm.loc[matched_t_idx, cov], df_psm.loc[matched_c_idx, cov])
        cov_smds.append({"covariate": cov, "smd_before": smd_before, "smd_after": smd_after})
    return pd.DataFrame(cov_smds)


def calculate_att(df_psm, matched_t_idx, matched_c_idx, outcome_var=OUTCOME_VAR):
    return df_psm.loc[matched_t_idx, outcome_var].mean() - df_psm.loc[matched_c_idx, outcome_var].mean()


def prepare_analysis_data(df_analysis, outcome_var=OUTCOME_VAR):
    exclude = [
        'treat', 'ups', 'virality_rss', 'score', 'num_comments', 'total_votes',
        'downs', 'subreddit_subscribers', 'Sentiment_pos_vader', 'Sentiment_neg_vader',
        'Sentiment_compound_vader', 'total_awards_received', 'upvote_ratio', 'gilded',
        'num_crossposts', 'is_video', 'created', 'approved_by', 'mod_reason_title',
        'top_awarded_type', 'report_reasons', 'removed_by', 'num_reports',
        'mod_reason_by', 'treatment_tags', 'view_count', 'banned_at_utc', 'likes',
        'banned_by', 'num_duplicates', 'discussion_type', 'mod_note', 'category',
        'approved_at_utc', 'pwls', 'wls', 'Frac_special', 'LIWC_Pronoun',
        'LIWC_Ppron', 'LIWC_Ipron', 'Frac_alpha', 'Frac_digits', 'Frac_whitespace',
        'is_self', 'created_utc', 'Num_unique_words', 'Num_unique_words_no_space'
    ]
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    covariates = [c for c in numeric_cols if c not in exclude]
    df_clean = df_analysis.dropna(subset=covariates + [outcome_var]).copy()
    for col in covariates:
        df_clean[col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
    return df_clean, covariates


def run_psm_analysis_detailed(treatment_var, df, covariates_list, exclusions=None,
                               support_q=SUPPORT_Q, caliper_mult=CALIPER_MULT, eps=EPS):
    covariates_to_use = filter_covariates(treatment_var, covariates_list, exclusions)
    
    df_work = df.copy()
    df_work, threshold = create_treatment_indicator(df_work, treatment_var)
    df_psm = df_work.dropna(subset=covariates_to_use + ['treat', OUTCOME_VAR]).copy()
    
    model = fit_propensity_model(df_psm, covariates_to_use)
    df_psm = compute_propensity_scores(df_psm, model, eps)
    
    treatment_df_full = df_psm[df_psm['treat'] == 1]
    control_df_full = df_psm[df_psm['treat'] == 0]
    
    support_low, support_high = calculate_common_support_bounds(df_psm, support_q)
    if support_low is None:
        return None
    
    n_before_support = len(df_psm)
    df_psm = trim_to_common_support(df_psm, support_low, support_high)
    n_after_support = len(df_psm)
    
    treatment_df = df_psm[df_psm['treat'] == 1]
    control_df = df_psm[df_psm['treat'] == 0]
    
    caliper_logit = calculate_caliper(df_psm, caliper_mult, CALIPER)
    matched_t_idx, matched_c_idx = perform_nearest_neighbor_matching(
        treatment_df, control_df, caliper_logit, score_col="logit_ps"
    )
    
    cov_balance = calculate_covariate_balance(
        treatment_df_full, control_df_full, matched_t_idx, matched_c_idx, 
        covariates_to_use, df_psm
    )
    
    att = calculate_att(df_psm, matched_t_idx, matched_c_idx)
    
    return {
        "summary": {
            "treatment_var": treatment_var,
            "threshold_median": float(threshold),
            "support_q": float(support_q),
            "support_ps_low": float(support_low),
            "support_ps_high": float(support_high),
            "n_before_support": int(n_before_support),
            "n_after_support": int(n_after_support),
            "caliper_mult": float(caliper_mult),
            "caliper_logit": float(caliper_logit),
            "att": float(att),
            "n_matched": int(len(matched_t_idx)),
        },
        "covariate_balance": cov_balance
    }


def run_psm_with_percentile_support(treatment_var, df, covariates_list, exclusions=None,
                                     outcome_var=OUTCOME_VAR, eps=EPS, support_q=SUPPORT_Q,
                                     caliper_mult=CALIPER_MULT, caliper_fallback=CALIPER):
    if df is None or covariates_list is None:
        raise ValueError("Pass df and covariates_list (e.g., df_clean and covariates).")
    
    if exclusions is None:
        exclusions = COVARIATE_EXCLUSIONS.get(treatment_var, [])
    
    covariates_to_use = filter_covariates(treatment_var, covariates_list, exclusions)
    
    df_work = df.copy()
    df_work, threshold = create_treatment_indicator(df_work, treatment_var)
    df_psm = df_work.dropna(subset=covariates_to_use + ["treat", outcome_var]).copy()
    
    model = fit_propensity_model(df_psm, covariates_to_use)
    df_psm = compute_propensity_scores(df_psm, model, eps)
    
    t = df_psm.loc[df_psm["treat"] == 1, "Propensity_score"]
    c = df_psm.loc[df_psm["treat"] == 0, "Propensity_score"]
    if len(t) == 0 or len(c) == 0:
        raise RuntimeError("No treated or control units after treatment definition.")
    
    support_low, support_high = calculate_common_support_bounds(df_psm, support_q)
    if support_low is None:
        raise RuntimeError("Could not calculate common support bounds.")
    
    n_before = len(df_psm)
    df_psm = trim_to_common_support(df_psm, support_low, support_high)
    n_after = len(df_psm)
    
    treatment_df = df_psm[df_psm["treat"] == 1]
    control_df = df_psm[df_psm["treat"] == 0]
    if len(treatment_df) == 0 or len(control_df) == 0:
        raise RuntimeError("No treated or control units after common-support trimming.")
    
    caliper = calculate_caliper(df_psm, caliper_mult, caliper_fallback)
    matched_t_idx, matched_c_idx = perform_nearest_neighbor_matching(
        treatment_df, control_df, caliper, score_col="logit_ps"
    )
    
    if len(matched_t_idx) == 0:
        raise RuntimeError("No matches found within caliper after trimming.")
    
    cov_balance = calculate_covariate_balance(
        treatment_df, control_df, matched_t_idx, matched_c_idx,
        covariates_to_use + ["logit_ps"], df_psm
    )
    
    att = calculate_att(df_psm, matched_t_idx, matched_c_idx, outcome_var)
    
    return {
        "summary": {
            "treatment_var": treatment_var,
            "threshold_median": float(threshold),
            "support_q": float(support_q),
            "support_ps_low": float(support_low),
            "support_ps_high": float(support_high),
            "n_before_support": int(n_before),
            "n_after_support": int(n_after),
            "caliper_mult": float(caliper_mult),
            "caliper_logit": float(caliper),
            "n_treated_after_support": int(len(treatment_df)),
            "n_controls_after_support": int(len(control_df)),
            "n_matched": int(len(matched_t_idx)),
            "att": float(att),
        },
        "covariate_balance": cov_balance,
        "matched_indices": {"treated": matched_t_idx, "control": matched_c_idx},
    }


def save_analysis_results(result, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    with gzip.open(out_path, "wb") as f:
        pd.to_pickle(result, f)
    return out_path


def print_top_covariates_by_smd(cov_balance, n=10, sort_by='smd_after'):
    top = cov_balance.sort_values(sort_by, key=lambda s: s.abs(), ascending=False).head(n)
    print(f"\nTop {n} by |{sort_by}|:")
    print(top)


def print_analysis_summary(summary_dict):
    print("\nPSM results:")
    for k, v in summary_dict.items():
        print(f"{k}: {v}")


def print_balance_statistics(cov_balance):
    worst_before = cov_balance['smd_before'].abs().max()
    worst_after = cov_balance['smd_after'].abs().max()
    mean_before = cov_balance['smd_before'].abs().mean()
    mean_after = cov_balance['smd_after'].abs().mean()
    
    print(f"\nWorst |SMD|: {worst_before:.3f} -> {worst_after:.3f}")
    print(f"Mean |SMD|: {mean_before:.3f} -> {mean_after:.3f}")

