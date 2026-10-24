import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

class InsuranceSegmenter:
    """
    Class for segmenting insurance customers to create specialized models.
    """
    
    def __init__(self):
        """Initialize the segmenter."""
        self.segmentation_model = None
        self.segment_features = None
        self.n_clusters = None
        self.scaler = StandardScaler()
        
    def create_segments(self, X, y, method='kmeans', n_clusters=4, features=None):
        """
        Create customer segments using clustering.
        
        Args:
            X: Features DataFrame
            y: Target variable (log-transformed premiums)
            method: Clustering method ('kmeans' currently supported)
            n_clusters: Number of clusters to create
            features: List of features to use for segmentation (None for all)
            
        Returns:
            Array of segment IDs for each sample
        """
        # Determine features to use for segmentation
        if features is None:
            # Use all numeric features by default
            features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        self.segment_features = features
        X_segment = X[features].copy()
        
        # Scale features for better clustering
        X_scaled = self.scaler.fit_transform(X_segment)
        
        # Create segments based on specified method
        if method == 'kmeans':
            self.segmentation_model = KMeans(n_clusters=n_clusters, random_state=42)
            segments = self.segmentation_model.fit_predict(X_scaled)
            self.n_clusters = n_clusters
        else:
            raise ValueError(f"Segmentation method '{method}' not supported")
        
        # Add segment information to data
        X['Segment'] = segments
        
        # Print segment statistics
        self._print_segment_stats(X, y, segments)
        
        return segments
    
    def assign_segment(self, X):
        """
        Assign segments to new data points.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of segment IDs
        """
        if self.segmentation_model is None:
            raise ValueError("Segmentation model not trained. Call create_segments first.")
        
        # Extract relevant features
        X_segment = X[self.segment_features].copy()
        
        # Scale features
        X_scaled = self.scaler.transform(X_segment)
        
        # Assign segments
        segments = self.segmentation_model.predict(X_scaled)
        
        return segments
    
    def _print_segment_stats(self, X, y, segments):
        """Print statistics for each segment."""
        for i in range(self.n_clusters):
            segment_mask = segments == i
            segment_count = segment_mask.sum()
            segment_pct = segment_count / len(X) * 100
            avg_premium = np.exp(y[segment_mask].mean()) - 1  # Convert back from log space
            
            print(f"Segment {i}: {segment_count} samples ({segment_pct:.1f}%), Avg Premium: ${avg_premium:.2f}")
    
    def visualize_segments(self, X, segments, features=None, n_components=2):
        """
        Visualize segments using PCA for dimensionality reduction.
        
        Args:
            X: Features DataFrame
            segments: Array of segment IDs
            features: Features to use for visualization (None for all segment features)
            n_components: Number of PCA components for visualization
        """
        # Use segment features if none specified
        if features is None:
            features = self.segment_features
        
        # Extract features and scale
        X_vis = X[features].copy()
        X_scaled = self.scaler.transform(X_vis)
        
        # Apply PCA for visualization
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create segment visualization
        plt.figure(figsize=(10, 8))
        
        # Plot each segment
        for segment_id in range(self.n_clusters):
            mask = segments == segment_id
            plt.scatter(
                X_pca[mask, 0], 
                X_pca[mask, 1],
                label=f'Segment {segment_id}',
                alpha=0.7
            )
        
        # Add labels and legend
        plt.title('Customer Segments (PCA Visualization)')
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        return plt.gcf()
