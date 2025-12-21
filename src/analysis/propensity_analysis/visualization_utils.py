"""
Plotting and printing functions for PSM results.
"""

import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_treatment_effects(df):
    order_map = {"Median": 0, "20th": 1, "40th": 2, "50th": 3, "60th": 4, "80th": 5}
    
    quantile_df = df[df["threshold_label_str"] != "Median"].copy()
    
    if len(quantile_df) == 0:
        print("No data to plot")
        return
    
    quantile_df = quantile_df.sort_values(["treatment_var_str", "threshold_order"])
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    pivot_att = quantile_df.pivot(
        index="treatment_var_str", columns="threshold_label_str", values="att"
    )
    pivot_att = pivot_att.reindex(
        columns=sorted(pivot_att.columns, key=lambda c: order_map.get(c, 99))
    )
    im1 = ax1.imshow(pivot_att.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
    ax1.set_yticks(range(len(pivot_att.index)))
    ax1.set_yticklabels(pivot_att.index, fontsize=9)
    ax1.set_xticks(range(len(pivot_att.columns)))
    ax1.set_xticklabels(pivot_att.columns)
    ax1.set_xlabel("Threshold Quantile", fontsize=11)
    ax1.set_title("Treatment Effect (ATT) by Feature and Threshold", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=ax1, label="ATT")
    
    ax2 = fig.add_subplot(gs[0, 2])
    pivot_top1 = quantile_df.pivot(
        index="treatment_var_str", columns="threshold_label_str", values="att_top1pct"
    )
    pivot_top1 = pivot_top1.reindex(
        columns=sorted(pivot_top1.columns, key=lambda c: order_map.get(c, 99))
    )
    im2 = ax2.imshow(pivot_top1.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
    ax2.set_yticks(range(len(pivot_top1.index)))
    ax2.set_yticklabels(pivot_top1.index, fontsize=9)
    ax2.set_xticks(range(len(pivot_top1.columns)))
    ax2.set_xticklabels(pivot_top1.columns)
    ax2.set_xlabel("Threshold Quantile", fontsize=11)
    ax2.set_title("ATT (Top 1% Viral) by Threshold", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=ax2, label="ATT (Top 1%)")
    
    ax3 = fig.add_subplot(gs[1, 0])
    strongest_impact = (
        quantile_df.groupby("treatment_var_str")["att"]
        .apply(lambda x: x.iloc[np.argmax(np.abs(x))])
        .sort_values())
    colors = ["red" if x < 0 else "green" for x in strongest_impact.values]
    ax3.barh(range(len(strongest_impact)), strongest_impact.values, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(strongest_impact)))
    ax3.set_yticklabels(strongest_impact.index, fontsize=9)
    ax3.axvline(0, color="black", linestyle="--", linewidth=1)
    ax3.set_xlabel("Strongest Observed ATT (Peak Split)", fontsize=11)
    ax3.set_title("Feature Importance: Strongest Impact", fontsize=12, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    avg_top1 = quantile_df.groupby("treatment_var_str")["att_top1pct"].mean().sort_values()
    colors_top1 = ["red" if (np.isfinite(x) and x < 0) else "green" for x in avg_top1.values]
    ax4.barh(range(len(avg_top1)), avg_top1.values, color=colors_top1, alpha=0.7)
    ax4.set_yticks(range(len(avg_top1)))
    ax4.set_yticklabels(avg_top1.index, fontsize=9)
    ax4.axvline(0, color="black", linestyle="--", linewidth=1)
    ax4.set_xlabel("Average ATT (Top 1%)", fontsize=11)
    ax4.set_title("What Matters for Highly Viral Posts", fontsize=12, fontweight="bold")
    ax4.grid(axis="x", alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 2])
    avg_att_all = quantile_df.groupby("treatment_var_str")["att"].mean()
    avg_top1_all = quantile_df.groupby("treatment_var_str")["att_top1pct"].mean()
    ax5.scatter(avg_att_all, avg_top1_all, s=100, alpha=0.6)
    for var in avg_att_all.index:
        ax5.annotate(var, (avg_att_all[var], avg_top1_all[var]), fontsize=8, ha="center", va="bottom")
    ax5.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax5.axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax5.set_xlabel("Average ATT (Overall)", fontsize=11)
    ax5.set_ylabel("Average ATT (Top 1% Viral)", fontsize=11)
    ax5.set_title("Overall vs Top 1% Viral Effects", fontsize=12, fontweight="bold")
    ax5.grid(alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 1:])
    top_vars = avg_att_all.abs().nlargest(5).index
    for var in top_vars:
        vd = quantile_df[quantile_df["treatment_var_str"] == var].sort_values("threshold_order")
        ax7.plot(vd["threshold_label_str"], vd["att"], marker="o", label=var, linewidth=2)
    ax7.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax7.set_xlabel("Threshold Quantile", fontsize=11)
    ax7.set_ylabel("ATT", fontsize=11)
    ax7.set_title("Treatment Effects Across Thresholds (Top 5 Features)", fontsize=12, fontweight="bold")
    ax7.legend(fontsize=8, loc="best")
    ax7.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_summary_table(df):
    print("\n" + "=" * 140)
    print("Treatment Effects Across All Variables and Thresholds")
    print("=" * 140)
    print(
        f"{'Treatment Variable':<25} {'Threshold':<12} {'N Treated':<12} {'N Control':<12} {'N Matched':<12} "
        f"{'Match Rate (%)':<15} {'ATT':<12} {'ATT (Top 1%)':<15} {'SMD before':<12} {'SMD after':<12}"
    )
    print("-" * 140)
    
    df_sorted = df.sort_values(["treatment_var_str", "threshold_order"])
    
    for _, r in df_sorted.iterrows():
        n_t = float(r["n_treated"])
        n_m = float(r["n_matched"])
        match_rate = (n_m / n_t * 100.0) if n_t else np.nan
        att_top1 = r["att_top1pct"]
        att_top1_str = f"{att_top1:.4f}" if np.isfinite(att_top1) else "N/A"
        
        print(
            f"{str(r['treatment_var_str']):<25} {str(r['threshold_label_str']):<12} "
            f"{int(r['n_treated']):<12,} {int(r['n_control']):<12,} {int(r['n_matched']):<12,} "
            f"{match_rate:<15.1f} {float(r['att']):<12.4f} {att_top1_str:<15} "
            f"{float(r['balance_smd_before']):<12.4f} {float(r['balance_smd_after']):<12.4f}"
        )
    
    print("=" * 140)
    print(f"\n{len(df):,} analyses, {df['treatment_var_str'].nunique()} variables")


def export_sorted_covariate_balance(input_path, output_folder, csv_filename="num_characters_covariate_balance.csv"):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
    
    with gzip.open(input_path, "rb") as f:
        result = pd.read_pickle(f)
    
    df_balance = result.get('covariate_balance')
    if df_balance is None or df_balance.empty:
        print("No covariate balance data found")
        return
    
    df_export = df_balance.copy()
    df_export.columns = ['variable', 'smd before matching', 'smd after matching']
    
    df_export['asmd before matching'] = df_export['smd before matching'].abs()
    df_export['asmd after matching'] = df_export['smd after matching'].abs()
    
    df_export = df_export[['variable', 'asmd before matching', 'asmd after matching']]
    df_export = df_export.sort_values(by='asmd before matching', ascending=False)
    
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, csv_filename)
    df_export.to_csv(csv_path, index=False)
    
    print(f"Exported to {csv_path}")
    print(f"\nTop 10 by aSMD before:")
    print(df_export.head(10))


def plot_covariate_balance(csv_path, top_n=10):
    df = pd.read_csv(csv_path)
    
    top = (
        df.sort_values("asmd before matching", ascending=False)
          .head(top_n)
          .copy()
    )
    
    y = np.arange(len(top))
    bar_h = 0.4
    
    plt.figure(figsize=(12, max(6, 0.35 * len(top))))
    plt.barh(y - bar_h/2, top["asmd before matching"].values, height=bar_h, label="Before (aSMD)")
    plt.barh(y + bar_h/2, top["asmd after matching"].values, height=bar_h, label="After (aSMD)")
    
    plt.yticks(y, top["variable"].astype(str).values)
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute Standardized Mean Difference (aSMD)")
    plt.title(f"Top {top_n} Covariates by aSMD Before Matching (Before vs After)")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.show()

