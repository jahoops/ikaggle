import pandas as pd
import numpy as np
from pathlib import Path
import logging

class EnsembleBuilder:
    def __init__(self, input_dir='inputs', output_dir='outputs'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def load_predictions(self):
        """Load all CSV prediction files from input directory"""
        predictions = {}
        for csv_file in self.input_dir.glob('*_estimates_*.csv'):
            # Extract model name and score from filename
            parts = csv_file.stem.split('_')
            model_name = parts[0]
            score = float(parts[-1])
            
            # Load predictions
            df = pd.read_csv(csv_file)
            predictions[model_name] = {
                'predictions': df['Premium Amount'],
                'score': score
            }
        return predictions

    def build_ensemble(self):
        """Build weighted ensemble from predictions"""
        # Load all predictions
        predictions = self.load_predictions()
        
        # Calculate weights based on scores
        scores = np.array([p['score'] for p in predictions.values()])
        weights = 1 / scores
        weights = weights / weights.sum()
        
        # Get IDs from first prediction file
        first_pred = next(iter(predictions.values()))['predictions']
        ensemble_df = pd.DataFrame({'id': range(len(first_pred))})
        
        # Calculate weighted average
        weighted_preds = np.zeros(len(first_pred))
        for (model, pred_dict), weight in zip(predictions.items(), weights):
            weighted_preds += pred_dict['predictions'] * weight
            
        ensemble_df['Premium Amount'] = weighted_preds.round(3)
        
        # Save ensemble predictions
        output_path = self.output_dir / 'ensemble_predictions.csv'
        ensemble_df.to_csv(output_path, index=False)
        return ensemble_df

if __name__ == "__main__":
    ensemble = EnsembleBuilder()
    ensemble.build_ensemble()