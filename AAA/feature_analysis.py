import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from scipy.stats import spearmanr

def analyze_target_distribution(data: pd.DataFrame, output_dir: Path):
    """Analyze PremiumAmount distribution and identify potential issues"""
    plt.figure(figsize=(12, 6))
    
    # Plot original and log-transformed distributions
    plt.subplot(1, 2, 1)
    sns.histplot(data['PremiumAmount'], bins=50)
    plt.title('Original PremiumAmount Distribution')
    plt.xlabel('PremiumAmount')
    
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(data['PremiumAmount']), bins=50)
    plt.title('Log-transformed PremiumAmount')
    plt.xlabel('log(PremiumAmount + 1)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'target_distribution.png')
    plt.close()
    
    # Calculate distribution statistics
    stats = {
        'mean': data['PremiumAmount'].mean(),
        'median': data['PremiumAmount'].median(),
        'skew': data['PremiumAmount'].skew(),
        'zeros': (data['PremiumAmount'] == 0).sum(),
        'negative': (data['PremiumAmount'] < 0).sum()
    }
    
    return stats

def analyze_feature_correlations(data: pd.DataFrame, target: str, output_dir: Path):
    """Analyze feature correlations with target"""
    correlations = []
    
    for col in data.columns:
        if col != target and col != 'id':
            spearman_corr, _ = spearmanr(data[col], data[target])
            correlations.append({
                'feature': col,
                'spearman_correlation': spearman_corr
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('spearman_correlation', key=abs, ascending=False)
    
    return corr_df

def main():
    # Load all available data
    print("Loading data...")
    data_dir = Path('/home/USER/Projects/ikaggle/data')
    train_data = pd.read_csv(data_dir / 'train.csv')
    validate_data = pd.read_csv(data_dir / 'validate.csv')
    
    output_dir = Path('/home/USER/Projects/ikaggle/AAA/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze target distribution
    print("\nAnalyzing target distribution...")
    train_stats = analyze_target_distribution(train_data, output_dir)
    val_stats = analyze_target_distribution(validate_data, output_dir)
    
    print("\nTarget Statistics:")
    print("Training:")
    for k, v in train_stats.items():
        print(f"  {k}: {v:.2f}")
    print("\nValidation:")
    for k, v in val_stats.items():
        print(f"  {k}: {v:.2f}")
    
    # Analyze feature correlations
    print("\nAnalyzing feature correlations...")
    train_corr = analyze_feature_correlations(train_data, 'PremiumAmount', output_dir)
    val_corr = analyze_feature_correlations(validate_data, 'PremiumAmount', output_dir)
    
    print("\nTop 10 most correlated features:")
    print(train_corr.head(10))
    
    # Compare train/validation distributions
    print("\nComparing train/validation distributions...")
    for col in train_data.columns:
        if col not in ['id', 'PremiumAmount']:
            ks_stat = np.abs(train_data[col].mean() - validate_data[col].mean())
            print(f"{col}: mean diff = {ks_stat:.4f}")
    
    # Save analysis results
    train_corr.to_csv(output_dir / 'feature_correlations.csv', index=False)
    
    print(f"\nAnalysis results saved to: {output_dir}")

if __name__ == "__main__":
    main()
