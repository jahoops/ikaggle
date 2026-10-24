import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
import seaborn as sns
from .optimal_segments import OPTIMAL_SEGMENTS

class FeatureAnalyzer:
    """Analyzes features to determine optimal segmentation strategies"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.target = 'PremiumAmount'
        
    def get_optimal_segments(self, feature: str) -> Dict:
        """Get optimal segmentation configuration for a feature"""
        if feature not in OPTIMAL_SEGMENTS:
            raise ValueError(f"No optimal segments defined for {feature}")
        return OPTIMAL_SEGMENTS[feature]
    
    def analyze_feature_impact(self, feature: str, use_optimal: bool = True) -> pd.DataFrame:
        """Analyze how a feature impacts the target variable"""
        if feature not in self.data.columns:
            raise ValueError(f"Feature {feature} not found in data")
        
        if use_optimal:
            optimal = self.get_optimal_segments(feature)
            # Create bins using optimal boundaries
            bins = pd.cut(
                self.data[feature], 
                bins=optimal['boundaries'],
                labels=optimal['labels']
            )
        else:
            # Use original quantile-based binning
            bins = pd.qcut(self.data[feature], q=20, duplicates='drop')
        
        # Calculate statistics by feature value with observed=True
        stats = self.data.groupby(bins, observed=True)\
            .agg({
                self.target: ['count', 'mean', 'std', 'median'],
                feature: ['min', 'max']
            }).round(2)
        
        # Calculate RMSLE for each segment
        stats['segment_rmsle'] = stats.index.map(lambda bin_: 
            np.sqrt(mean_squared_log_error(
                self.data[self.target][bins == bin_],
                [stats.at[bin_, (self.target, 'mean')]] * stats.at[bin_, (self.target, 'count')]
            ))
        )
        
        return stats

    def plot_feature_segments(self, feature: str, use_optimal: bool = True):
        """Plot feature segmentation analysis"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
        
        # Plot 1: Premium Amount vs Feature with segment boundaries
        sns.scatterplot(data=self.data, x=feature, y=self.target, alpha=0.5, ax=ax1)
        
        if use_optimal:
            optimal = self.get_optimal_segments(feature)
            boundaries = optimal['boundaries']
            labels = optimal['labels']
            n_segments = len(labels)  # Get number of segments from optimal config
        else:
            n_segments = 6
            boundaries = np.linspace(
                self.data[feature].min(),
                self.data[feature].max(),
                n_segments + 1
            )
            labels = [f'Segment {i}' for i in range(n_segments)]
        
        # Add segment boundaries
        for boundary, label in zip(boundaries[:-1], labels):  # Note: using boundaries[:-1]
            ax1.axvline(x=boundary, color='r', linestyle='--', alpha=0.5)
            ax1.text(boundary, ax1.get_ylim()[1], label, 
                    rotation=45, ha='right')
        
        # Get stats for the bar plot
        stats = self.analyze_feature_impact(feature, use_optimal=use_optimal)
        
        # Prepare data for bar plot by flattening the multi-index
        plot_data = pd.DataFrame({
            'segment_min': stats[feature]['min'],
            'segment_max': stats[feature]['max'],
            'rmsle': stats['segment_rmsle']
        })
        
        # Plot 2: Segment RMSLE scores
        sns.barplot(
            data=plot_data,
            x=plot_data.index,  # Using segment index
            y='rmsle',
            ax=ax2
        )
        
        # Customize plots
        title = f'{feature} vs Premium Amount with '
        title += f'{n_segments} Optimal Segments' if use_optimal else f'{n_segments} Equal Segments'
        ax1.set_title(title)
        ax2.set_title('RMSLE Score by Segment')
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('RMSLE Score')
        
        plt.tight_layout()
        return fig