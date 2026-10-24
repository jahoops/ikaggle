from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from segmentexperiments import XGBoostSegmentExperiment, SVRSegmentExperiment

def run_experiments_for_meta(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> tuple:
    """Run both experiments and collect predictions for meta-selector"""
    
    # Initialize experiments
    output_base = Path('/home/USER/Projects/ikaggle/experiments/output/meta_selector')
    output_base.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    experiments = {
        'xgboost': XGBoostSegmentExperiment(
            output_dir=output_base / f'xgboost_{timestamp}',
            test_mode=False
        ),
        'svr': SVRSegmentExperiment(
            output_dir=output_base / f'svr_{timestamp}',
            test_mode=False
        )
    }
    
    # Container for predictions
    predictions = {
        'train': pd.DataFrame(index=train_data.index),
        'val': pd.DataFrame(index=val_data.index),
        'test': pd.DataFrame(index=test_data.index)
    }
    
    # Run experiments and collect predictions
    for name, experiment in experiments.items():
        print(f"\nRunning {name} experiment...")
        
        # Train model
        experiment.train(train_data)
        
        # Get predictions for all datasets
        predictions['train'][f'{name}_pred'] = experiment.predict(train_data)['PremiumAmount']
        predictions['val'][f'{name}_pred'] = experiment.predict(val_data)['PremiumAmount']
        predictions['test'][f'{name}_pred'] = experiment.predict(test_data)['PremiumAmount']
        
        # Save individual experiment predictions
        for dataset_type, preds in predictions.items():
            experiment.save_predictions(
                preds[[f'{name}_pred']],
                f'{dataset_type}_predictions'
            )
    
    # Add actual values where available
    for dataset_type, data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        if 'PremiumAmount' in data.columns:
            predictions[dataset_type]['actual'] = data['PremiumAmount']
    
    # Save combined predictions
    for dataset_type, preds in predictions.items():
        output_path = output_base / f'{dataset_type}_combined_predictions_{timestamp}.csv'
        preds.to_csv(output_path, index=True)
        print(f"\nSaved combined {dataset_type} predictions to: {output_path}")
    
    # Prepare feature values for meta-selector
    feature_values = {
        'train': train_data[experiments['xgboost'].feature_columns],
        'val': val_data[experiments['xgboost'].feature_columns],
        'test': test_data[experiments['xgboost'].feature_columns]
    }
    
    return predictions, feature_values

def main():
    # Load your data
    train_data = pd.read_csv('/home/USER/Projects/ikaggle/data/train.csv')
    val_data = pd.read_csv('/home/USER/Projects/ikaggle/data/validate.csv')
    test_data = pd.read_csv('/home/USER/Projects/ikaggle/data/test.csv')
    
    # Run experiments and get predictions
    predictions, feature_values = run_experiments_for_meta(
        train_data, val_data, test_data
    )
    
    # Now you can use these with meta_selector:
    from meta_selector import PredictionSelector
    
    selector = PredictionSelector(feature_importance_threshold=0.01)
    selector.train_selector(
        all_predictions=predictions['train'].drop(columns=['actual']),
        actual_values=predictions['train']['actual'],
        feature_values=feature_values['train']
    )
    
    print("\nMeta-selector training complete!")

if __name__ == "__main__":
    main()
