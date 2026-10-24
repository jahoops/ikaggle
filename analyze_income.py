import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from experiments.data_manager import data_manager
from scipy import stats

def analyze_income_distribution():
    # Load the data
    train_data = data_manager.finaltrain.copy()
    
    # Basic statistics
    print("\nAnnual Income Statistics:")
    print("-" * 50)
    print(f"Mean: {train_data['AnnualIncome'].mean():.2f}")
    print(f"Median: {train_data['AnnualIncome'].median():.2f}")
    print(f"Std Dev: {train_data['AnnualIncome'].std():.2f}")
    print(f"Min: {train_data['AnnualIncome'].min():.2f}")
    print(f"Max: {train_data['AnnualIncome'].max():.2f}")
    
    # Calculate percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nPercentile Distribution:")
    print("-" * 50)
    for p in percentiles:
        value = np.percentile(train_data['AnnualIncome'], p)
        print(f"{p}th percentile: {value:.2f}")
    
    # Find natural clusters using KMeans
    from sklearn.cluster import KMeans
    X = train_data['AnnualIncome'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    print("\nNatural Income Clusters (Centers):")
    print("-" * 50)
    for i, center in enumerate(centers):
        print(f"Cluster {i+1}: {center:.2f}")
    
    # Analyze relationship with PremiumAmount
    correlation = train_data['AnnualIncome'].corr(train_data['PremiumAmount'])
    print(f"\nCorrelation with Premium Amount: {correlation:.3f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Histogram
    plt.subplot(2, 1, 1)
    plt.hist(train_data['AnnualIncome'], bins=50)
    plt.title('Distribution of Annual Income')
    plt.xlabel('Annual Income')
    plt.ylabel('Frequency')
    
    # Add vertical lines for cluster centers
    for center in centers:
        plt.axvline(x=center, color='r', linestyle='--', alpha=0.5)
    
    # Scatter plot with Premium Amount
    plt.subplot(2, 1, 2)
    plt.scatter(train_data['AnnualIncome'], train_data['PremiumAmount'], alpha=0.5)
    plt.title('Annual Income vs Premium Amount')
    plt.xlabel('Annual Income')
    plt.ylabel('Premium Amount')
    
    plt.tight_layout()
    plt.savefig('income_analysis.png')
    plt.close()

if __name__ == "__main__":
    analyze_income_distribution()