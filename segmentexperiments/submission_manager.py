from pathlib import Path
from typing import Dict, Any
import pandas as pd

class SegmentSubmissionManager:
    """Manages segment-specific model predictions and submissions"""
    
    def __init__(self, predictions_dir: Path):
        """Initialize submission manager
        
        Args:
            predictions_dir: Directory where predictions will be stored
        """
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
    
    def save_predictions(self, 
                        predictions: pd.Series,
                        segment_name: str,
                        dataset_type: str) -> None:
        """Save predictions for a specific segment and dataset"""
        pred_path = self.predictions_dir / f"{segment_name}_{dataset_type}_pred.csv"
        predictions.to_csv(pred_path)