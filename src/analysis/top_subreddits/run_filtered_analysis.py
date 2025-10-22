import os
import shutil
from src.analysis.top_subreddits.top_subreddits_analysis import main as run_top_subreddits_analysis
from src.analysis.top_subreddits.analyze_pair_matrices import analyze_pair_matrices

def run_full_analysis():
    print("Running initial top_subreddits_analysis...")
    run_top_subreddits_analysis(output_dir="src/analysis/top_subreddits/charts")

    print("Running analyze_pair_matrices...")
    base_dir = 'src/analysis/top_subreddits/charts/liwc_corr_cov'
    csv_path = f"{base_dir}/all_corr.csv"
    output_dir = f"{base_dir}/analysis"
    features_to_keep, interesting_pairs = analyze_pair_matrices(csv_path, 'corr', output_dir)

    old_dir = 'src/analysis/top_subreddits/charts'
    new_old_dir = 'src/analysis/top_subreddits/charts_old'
    if os.path.exists(new_old_dir):
        shutil.rmtree(new_old_dir)
    shutil.move(old_dir, new_old_dir)
    print(f"Renamed {old_dir} to {new_old_dir}")

    print(f"Running top_subreddits_analysis with filtered features ({len(features_to_keep)} features)...")
    run_top_subreddits_analysis(features=features_to_keep, output_dir="src/analysis/top_subreddits/charts")

    print("Analysis complete. Results in src/analysis/top_subreddits/charts, old results in src/analysis/top_subreddits/charts_old")

if __name__ == "__main__":
    run_full_analysis()
