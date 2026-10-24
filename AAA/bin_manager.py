import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path

class BinManager:
    def __init__(self, min_bin_size=1000, chunk_size=100000):
        self.min_bin_size = min_bin_size
        self.chunk_size = chunk_size
        self.feature_bins = {}
        self._load_existing_bins()
    
    def _load_existing_bins(self):
        """Load pre-calculated optimal bins from binning_results.csv"""
        try:
            bin_path = Path('/home/USER/Projects/ikaggle/AAA/binning_output/binning_results.csv')
            if bin_path.exists():
                bin_df = pd.read_csv(bin_path, index_col=0)
                print(f"Loading pre-calculated bins for {len(bin_df)} features")
                
                for feature, row in bin_df.iterrows():
                    if pd.notna(row['splits']) and row['splits'] != '[]':
                        # Convert string representation of list to actual list
                        splits = eval(row['splits'])
                        if splits:  # Only store if we have valid splits
                            self.feature_bins[feature] = {
                                "edges": np.array([-np.inf] + splits + [np.inf]),
                                "n_bins": row['n_bins'],
                                "status": row['status']
                            }
                print(f"Loaded bins for {len(self.feature_bins)} features")
        except Exception as e:
            print(f"Warning: Could not load pre-calculated bins: {str(e)}")

    def create_bins(self, df: pd.DataFrame, features: List[str]):
        """Create bins using pre-calculated values or calculate new ones"""
        for feature in features:
            if feature in self.feature_bins:
                print(f"Using pre-calculated bins for {feature}")
                continue
                
            print(f"Calculating new bins for {feature}")
            # Get initial edges using a sample
            sample_size = min(100000, len(df))
            sample = df[feature].sample(n=sample_size, random_state=42)
            
            try:
                # Use existing bin edges if available in pre-calculated results
                if feature in self.feature_bins:
                    edges = self.feature_bins[feature]["edges"]
                else:
                    edges = pd.qcut(sample, q=5, duplicates='drop', retbins=True)[1]
                
                # Process in chunks to get counts
                total_counts = self._count_bins(df[feature], edges)
                
                # Merge bins if needed
                while total_counts.min() < self.min_bin_size and len(edges) > 2:
                    edges = self._merge_smallest_bin(edges, total_counts)
                    total_counts = self._count_bins(df[feature], edges)
                
                self.feature_bins[feature] = {
                    "edges": edges,
                    "n_bins": len(edges) - 1,
                    "status": "OPTIMAL"
                }
            except Exception as e:
                print(f"Warning: Could not create bins for {feature}: {str(e)}")
                continue
                
        return self.feature_bins

    def _count_bins(self, series: pd.Series, edges: np.ndarray) -> pd.Series:
        """Count bin populations in chunks"""
        total_counts = pd.Series(dtype=int)
        for chunk_start in range(0, len(series), self.chunk_size):
            chunk = series.iloc[chunk_start:chunk_start + self.chunk_size]
            chunk_binned = pd.cut(chunk, edges, include_lowest=True)
            chunk_counts = chunk_binned.value_counts()
            total_counts = total_counts.add(chunk_counts, fill_value=0)
        return total_counts

    def get_bin_edges(self, feature: str) -> np.ndarray:
        """Get bin edges for a feature"""
        return self.feature_bins.get(feature, {}).get("edges", None)

    def transform(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Transform features into their binned versions"""
        result = pd.DataFrame()
        for feature in features:
            if feature in self.feature_bins:
                edges = self.feature_bins[feature]["edges"]
                result[f"{feature}_binned"] = pd.cut(
                    df[feature], 
                    bins=edges,
                    labels=range(len(edges)-1),
                    include_lowest=True
                )
        return result

    def get_bin_info(self) -> Dict:
        """Get information about all bins"""
        return {
            feature: {
                "n_bins": info["n_bins"],
                "edges": info["edges"].tolist(),
                "status": info.get("status", "UNKNOWN")
            }
            for feature, info in self.feature_bins.items()
        }