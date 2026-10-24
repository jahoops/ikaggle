from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.cluster import KMeans

class SegmentOptimizer:
    """Optimizes segment boundaries for features"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.target = 'PremiumAmount'
        
    def optimize_segments(self, feature: str, 
                         min_segments: int = 3,
                         max_segments: int = 10) -> Dict:
        """Find optimal number of segments for a feature"""
        results = []
        
        for n_segments in range(min_segments, max_segments + 1):
            # Try different segmentation approaches
            methods = {
                'quantile': self._evaluate_quantile_segments,
                'kmeans': self._evaluate_kmeans_segments,
                'equal_width': self._evaluate_equal_width_segments
            }
            
            for method_name, method in methods.items():
                score, boundaries = method(feature, n_segments)
                results.append({
                    'feature': feature,
                    'n_segments': n_segments,
                    'method': method_name,
                    'rmsle': score,
                    'boundaries': boundaries
                })
        
        return pd.DataFrame(results)
    
    def _evaluate_quantile_segments(self, feature: str, n_segments: int) -> Tuple[float, List]:
        """Evaluate quantile-based segmentation"""
        boundaries = self.data[feature].quantile(
            np.linspace(0, 1, n_segments + 1)
        ).tolist()
        score = self._calculate_segment_score(feature, boundaries)
        return score, boundaries
    
    def _evaluate_kmeans_segments(self, feature: str, n_segments: int) -> Tuple[float, List]:
        """Evaluate K-means based segmentation"""
        kmeans = KMeans(n_clusters=n_segments, random_state=42)
        kmeans.fit(self.data[[feature]])
        centers = sorted(kmeans.cluster_centers_.flatten())
        boundaries = [self.data[feature].min()] + centers + [self.data[feature].max()]
        score = self._calculate_segment_score(feature, boundaries)
        return score, boundaries
    
    def _evaluate_equal_width_segments(self, feature: str, n_segments: int) -> Tuple[float, List]:
        """Evaluate equal-width segmentation"""
        feature_min = self.data[feature].min()
        feature_max = self.data[feature].max()
        step = (feature_max - feature_min) / n_segments
        
        # Create boundaries with equal widths
        boundaries = [feature_min + i * step for i in range(n_segments + 1)]
        
        # Calculate score for these boundaries
        score = self._calculate_segment_score(feature, boundaries)
        return score, boundaries
    
    def _calculate_segment_score(self, feature: str, boundaries: List) -> float:
        """Calculate overall RMSLE score for a segmentation"""
        total_error = 0
        total_samples = 0
        
        for i in range(len(boundaries) - 1):
            mask = (self.data[feature] >= boundaries[i]) & \
                   (self.data[feature] < boundaries[i + 1])
            segment = self.data[mask]
            
            if len(segment) > 0:
                pred = [segment[self.target].mean()] * len(segment)
                error = mean_squared_log_error(segment[self.target], pred)
                total_error += error * len(segment)
                total_samples += len(segment)
        
        return np.sqrt(total_error / total_samples) if total_samples > 0 else float('inf')