import pandas as pd
import numpy as np
from typing import List, Dict, Union
from bin_manager import BinManager

class BinEncoder:
    """Encode features into binary columns based on bin edges"""
    
    def __init__(self, bin_manager: BinManager):
        self.bin_manager = bin_manager
        self.feature_bins = {}
        self.binary_columns = []
    
    def fit(self, df: pd.DataFrame, features: List[str]) -> 'BinEncoder':
        """Learn the binary encoding for each feature"""
        self.feature_bins = {}
        self.binary_columns = []
        
        for feature in features:
            if feature in self.bin_manager.feature_bins:
                edges = self.bin_manager.feature_bins[feature]['edges']
                # Create binary column names for this feature
                bin_names = []
                for i in range(len(edges) - 1):
                    bin_name = f"{feature}_bin_{i}"
                    bin_names.append(bin_name)
                self.feature_bins[feature] = {
                    'edges': edges,
                    'bin_names': bin_names
                }
                self.binary_columns.extend(bin_names)
        
        return self
    
    def transform(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Transform features into binary columns"""
        result = pd.DataFrame()
        
        for feature in features:
            if feature in self.feature_bins:
                # Get bin edges and names
                edges = self.feature_bins[feature]['edges']
                bin_names = self.feature_bins[feature]['bin_names']
                
                # Cut the data into bins
                binned = pd.cut(df[feature], 
                              bins=edges,
                              labels=False,
                              include_lowest=True)
                
                # Create binary columns
                for i, bin_name in enumerate(bin_names):
                    result[bin_name] = (binned == i).astype(int)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df, features).transform(df, features)
    
    def get_binary_columns(self) -> List[str]:
        """Get list of binary column names"""
        return self.binary_columns
    
    def get_feature_bins(self) -> Dict:
        """Get feature binning information"""
        return self.feature_bins
