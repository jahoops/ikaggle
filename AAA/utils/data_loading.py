import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Optional, Tuple
from ikaggle.AAA.config.feature_config import FEATURE_CONFIGS
from ikaggle.AAA.config.paths import DATA_PATHS


class DataLoader:
    """Unified data loading and preprocessing class"""
    
    def __init__(self, data_paths: Dict[str, str] = DATA_PATHS):
        self.data_paths = data_paths
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._genome_template = None
        self._total_gene_length = None
        self.target_column = 'PremiumAmount'
        self.excluded_columns = ['id', 'PremiumAmount']  # Add any other columns to exclude
        self.id_column = 'id'  # Add ID column name as class attribute

    def load_dataset(self, dataset_type='train') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Load and preprocess a specific dataset"""
        # Check file existence
        if not os.path.exists(self.data_paths[dataset_type]):
            raise FileNotFoundError(f"Dataset file not found: {self.data_paths[dataset_type]}")
        
        # Load data
        self.logger.info(f"Loading {dataset_type} dataset from: {self.data_paths[dataset_type]}")
        df = pd.read_csv(self.data_paths[dataset_type])
        self.logger.info(f"Available columns: {list(df.columns)}")
        
        # Always store ID if present (before any column dropping)
        id_series = None
        if self.id_column in df.columns:
            id_series = df[self.id_column].copy()
            df = df.drop(columns=[self.id_column], errors='ignore')
        
        # For test set, keep ID column for submission
        if dataset_type == 'test':
            # Only drop columns that actually exist
            cols_to_drop = [col for col in self.excluded_columns if col in df.columns]
            features = df.drop(columns=cols_to_drop, errors='ignore')
            return features, id_series
            
        # For training data
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
            
        # Extract target and drop columns that exist
        y = df[self.target_column]
        cols_to_drop = [col for col in self.excluded_columns if col in df.columns]
        X = df.drop(columns=cols_to_drop + [self.target_column], errors='ignore')
        
        # Convert categorical columns to ordinal format
        X = self._encode_categorical_features(X)
        
        # Sort both X and y by index to ensure alignment
        X = X.sort_index()
        y = y.sort_index()
        
        # Verify alignment
        if not X.index.equals(y.index):
            raise ValueError(
                f"Index mismatch after loading!\n"
                f"X indices: {X.index[:5]}\n"
                f"y indices: {y.index[:5]}"
            )
            
        self.logger.info(f"Verified index alignment for {len(X)} samples")
        
        # Log loading success
        self.logger.info(f"Loaded {dataset_type} dataset: {len(X)} samples, {len(X.columns)} features")
        self.logger.info(f"PremiumAmount range: min={y.min():.2f}, max={y.max():.2f}")
        
        return X, y

    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical features to numerical values"""
        X = X.copy()
        for col in X.columns:
            if col in FEATURE_CONFIGS:
                config = FEATURE_CONFIGS[col]
                if config['type'] == 'ordinal':
                    X[col] = X[col].map(config['mapping']).fillna(0.0)
                elif config['type'] == 'boolean':
                    X[col] = X[col].map(config['mapping']).fillna(0)
        return X

    @property
    def genome_template(self):
        """Lazy loading of genome template"""
        if self._genome_template is None:
            self._genome_template = self._create_genome_template()
        return self._genome_template

    @property
    def total_gene_length(self):
        """Lazy calculation of total gene length"""
        if self._total_gene_length is None:
            self._total_gene_length = sum(x[1] for x in self.genome_template)
        return self._total_gene_length

    def _create_genome_template(self):
        """Create genome template based on feature configurations"""
        template = []
        
        # Add base features
        for feature in FEATURE_CONFIGS.keys():
            template.append((feature, 1))  # Activation bit
            config = FEATURE_CONFIGS[feature]
            
            if config['type'] == 'variable':
                gene_length = max(1, int(np.ceil(np.log2(config['norm_scale']))))
                template.append((f"{feature}_value", gene_length))
            elif config['type'] == 'binned':
                n_bins = len(config['bins'])
                gene_length = max(1, int(np.ceil(np.log2(n_bins))))
                template.append((f"{feature}_bin", gene_length))
        
        # Add combination slots
        template.extend(self._create_combination_template())
        
        return template

    def _create_combination_template(self):
        """Create template for feature combinations"""
        template = []
        # Add pair combinations
        for i in range(200):
            template.extend([
                (f"pair_{i}_active", 1),
                (f"pair_{i}_f1", 8),
                (f"pair_{i}_f2", 8),
                (f"pair_{i}_op", 3),
                (f"pair_{i}_raw", 1)
            ])
        
        # Add triplet combinations
        for i in range(50):
            template.extend([
                (f"triple_{i}_active", 1),
                (f"triple_{i}_f1", 8),
                (f"triple_{i}_f2", 8),
                (f"triple_{i}_f3", 8),
                (f"triple_{i}_op1", 3),
                (f"triple_{i}_op2", 3),
                (f"triple_{i}_raw", 1)
            ])
        
        return template

# Create singleton instance
data_loader = DataLoader()

GENOME_TEMPLATE = data_loader.genome_template
TOTAL_GENE_LENGTH = data_loader.total_gene_length