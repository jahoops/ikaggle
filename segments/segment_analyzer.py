import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from scipy.stats import entropy

class SegmentAnalyzer:
    """Analyzes features to determine optimal segmentation strategies"""
    
    def __init__(self, data, target='PremiumAmount'):
        self.data = data
        self.target = target
        
    def analyze_feature_segments(self, feature_name, n_segments=6):
        """Analyze the predictive power of feature segments"""
        # Create segments using quantiles for more balanced distributions
        segments = pd.qcut(self.data[feature_name], q=n_segments, duplicates='drop')
        
        # Calculate statistics for each segment
        segment_stats = []
        for segment_name, segment_data in self.data.groupby(segments):
            stats = {
                'feature': feature_name,
                'segment_range': f"{segment_name.left:.2f}-{segment_name.right:.2f}",
                'count': len(segment_data),
                'mean_premium': segment_data[self.target].mean(),
                'std_premium': segment_data[self.target].std(),
                'median_premium': segment_data[self.target].median(),
                'rmsle': np.sqrt(mean_squared_log_error(
                    segment_data[self.target], 
                    [segment_data[self.target].mean()] * len(segment_data)
                ))
            }
            segment_stats.append(stats)
        
        return pd.DataFrame(segment_stats)