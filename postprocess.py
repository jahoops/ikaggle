import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from utils import feature_cols
from experiments.data_manager import data_manager
import logging

def postprocess():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base_path = "data/processed"

    cleaned_train = data_manager.loadtrain.copy()
    cleaned_validate = data_manager.loadvalidate.copy()
    cleaned_test = data_manager.loadtest.copy()

    feature_files = {
        "train": "feature_train.csv",
        "validate": "feature_validate.csv",
        "test": "feature_test.csv"
    }
    nn_files = {
        "train": "nn_train.csv",
        "validate": "nn_validate.csv",
        "test": "nn_test.csv"
    }
    final_files = {
        "train": "final_train.csv",
        "validate": "final_validate.csv",
        "test": "final_test.csv"
    }
    # Include required columns
    required_columns = set(["id", "PremiumAmount"])
    keep_cols = set(feature_cols).union(required_columns)

    logger.info("Starting post-processing...")

    # Iterate through the datasets and process them
    for split, df in [('train', cleaned_train), ('validate', cleaned_validate), ('test', cleaned_test)]:
        if df is None:
            logger.error(f"{split.upper()} dataset is empty. Skipping post-processing.")
            raise ValueError(f"{split.upper()} dataset is empty.")
        
        keep_split = [c for c in df.columns if c in keep_cols]
        df = df[keep_split]

        missing_cols_split = keep_cols - set(keep_split)
        if missing_cols_split:
            logger.warning(f"{split.upper()} is missing columns: {missing_cols_split}. Adding them with default values.")
            for col in missing_cols_split:
                if col == "PremiumAmount" and split == "test":
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(0)

        df = df.copy()

        # Verify columns
        if sorted(df.columns) != sorted(keep_cols):
            logger.error(f"{split.upper()} columns mismatch. Expected: {sorted(keep_cols)}, Got: {sorted(df.columns)}")
            raise ValueError(f"Columns mismatch in {split.upper()} dataset.")

        # Save processed features
        output_path = os.path.join(base_path, final_files[split])
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed final files to {output_path}")

if __name__ == "__main__":
    postprocess()
