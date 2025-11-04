import os
import shutil
from src.analysis.top_subreddits.top_subreddits_analysis import main as run_top_subreddits_analysis
from src.analysis.top_subreddits.analyze_pair_matrices import analyze_pair_matrices

def run_full_analysis():
    charts_dir = "figures/top_subreddits_analysis_charts"
    results_dir = "src/analysis/top_subreddits/results"
    
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("Running initial top_subreddits_analysis...")
    run_top_subreddits_analysis(output_dir=charts_dir)

    print("Running analyze_pair_matrices...")
    csv_path = f"{charts_dir}/liwc_corr_cov/all_corr.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Check if top_subreddits_analysis ran successfully.")
        return
    
    print(f"Analyzing correlation matrices and saving results to {results_dir}...")
    features_to_keep, interesting_pairs = analyze_pair_matrices(csv_path, 'corr', results_dir)

    print(f"Backing up charts to {charts_dir}/old...")
    old_backup_dir = f"{charts_dir}/old"
    if os.path.exists(old_backup_dir):
        shutil.rmtree(old_backup_dir)
    os.makedirs(old_backup_dir, exist_ok=True)
    
    for item in os.listdir(charts_dir):
        item_path = os.path.join(charts_dir, item)
        if item != 'old' and os.path.exists(item_path):
            dest_path = os.path.join(old_backup_dir, item)
            if os.path.isdir(item_path):
                shutil.copytree(item_path, dest_path)
                shutil.rmtree(item_path)
            else:
                shutil.copy2(item_path, dest_path)
                os.remove(item_path)
    print(f"Backed up old charts to {old_backup_dir}")

    print(f"Running top_subreddits_analysis with filtered features ({len(features_to_keep)} features)...")
    run_top_subreddits_analysis(features=features_to_keep, output_dir=charts_dir)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Charts saved to: {charts_dir}")
    print(f"Analysis results saved to: {results_dir}")
    print(f"Old charts backed up to: {old_backup_dir}")

if __name__ == "__main__":
    run_full_analysis()
