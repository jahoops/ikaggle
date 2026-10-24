import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def combine_feature_importance_methods(segmenter, X, y, methods=['random_forest', 'correlation'], n_features=15):
    """
    Combine multiple feature importance methods to get more robust feature selection
    
    Args:
        segmenter: InsuranceSegmenter instance
        X: Feature DataFrame
        y: Target values
        methods: List of methods to use
        n_features: Number of top features to select
        
    Returns:
        Combined list of most important features
    """
    all_importances = {}
    combined_ranks = {}
    
    # Calculate importance using each method
    for method in methods:
        print(f"\nAnalyzing feature importance using {method}...")
        result = segmenter.analyze_feature_importance(X, y, method=method, n_features=n_features)
        all_importances[method] = result
        
        # Store ranks for each feature
        for i, (feature, importance) in enumerate(result[['Feature', 'Importance']].values):
            if feature not in combined_ranks:
                combined_ranks[feature] = []
            combined_ranks[feature].append(i + 1)  # +1 because ranks start at 1
    
    # Calculate average rank for each feature
    avg_ranks = {feature: np.mean(ranks) for feature, ranks in combined_ranks.items()}
    
    # Sort features by average rank
    sorted_features = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    # Create visualization of combined feature importance
    plt.figure(figsize=(12, 8))
    feature_names = [f[0] for f in sorted_features[:n_features]]
    feature_scores = [1/f[1] for f in sorted_features[:n_features]]  # Invert because lower rank is better
    
    sns.barplot(x=feature_scores, y=feature_names)
    plt.title('Top Features (Combined Methods)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f"{segmenter.results_dir}/feature_importance_combined.png")
    plt.close()
    
    # Set the top features in the segmenter
    segmenter.top_features = feature_names
    
    # Return the list of top features
    return feature_names
