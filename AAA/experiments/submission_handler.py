import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

def create_submission(individual, predictions, test_ids, generation, individual_id=None):
    """Create submission file with predictions and verify its creation"""
    try:
        # Input validation
        if predictions is None or len(predictions) == 0:
            raise ValueError("No predictions provided")
            
        if test_ids is None or len(test_ids) == 0:
            raise ValueError("No test IDs provided")
            
        # Force predictions to numpy array
        predictions = np.array(predictions).ravel()
        test_ids = np.array(test_ids)
        
        if len(predictions) != len(test_ids):
            raise ValueError(f"Length mismatch: predictions ({len(predictions)}) != test_ids ({len(test_ids)})")
        
        # Create submission directory
        submission_dir = '/home/USER/Projects/ikaggle/AAA/submissions'
        os.makedirs(submission_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fitness = getattr(individual, 'last_fitness', float('inf'))
        csv_filename = f"submission_gen{generation:04d}_{timestamp}_{fitness:.4f}.csv"
        csv_path = os.path.join(submission_dir, csv_filename)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'ID': test_ids,
            'PremiumAmount': predictions
        })
        
        # Save submission with explicit options
        submission_df.to_csv(csv_path, index=False, float_format='%.2f')
        
        # Verify file creation and content
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Failed to create submission file: {csv_path}")
            
        file_size = os.path.getsize(csv_path)
        if file_size == 0:
            raise ValueError(f"Created submission file is empty: {csv_path}")
            
        # Verify content by reading back
        verify_df = pd.read_csv(csv_path)
        if len(verify_df) != len(test_ids):
            raise ValueError(f"Submission file has wrong number of rows: {len(verify_df)} vs expected {len(test_ids)}")
            
        if not all(col in verify_df.columns for col in ['ID', 'PremiumAmount']):
            raise ValueError(f"Missing required columns. Found: {verify_df.columns.tolist()}")
        
        # Save metadata
        metadata = {
            'generation': generation,
            'timestamp': timestamp,
            'fitness': float(fitness),
            'individual_id': individual_id or 'unknown',
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'median': float(np.median(predictions))
            },
            'file_info': {
                'path': csv_path,
                'size': file_size,
                'rows': len(verify_df)
            }
        }
        
        metadata_path = csv_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nSubmission created and verified:")
        print(f"Path: {csv_path}")
        print(f"Size: {file_size} bytes")
        print(f"Rows: {len(verify_df)}")
        print(f"Value range: {metadata['prediction_stats']['min']:.2f} - {metadata['prediction_stats']['max']:.2f}")
        
        return csv_path
        
    except Exception as e:
        print(f"Error creating submission: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Stop program on submission creation failure
