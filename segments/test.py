import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

# Direct imports without relative paths
from segmentexperiments import XGBoostSegmentExperiment
from segmentexperiments import SVRSegmentExperiment
from experiments.data_manager import DataManager
from segmentexperiments import NeuralNetSegmentExperiment

def run_experiment(
    experiment_class, 
    name: str, 
    sample_size: int = 96000, 
    notes: str = None
) -> dict:
    """Run a segment experiment with proper sample sizes"""
    output_dir = Path(__file__).parent / name
    output_dir.mkdir(exist_ok=True)
    
    experiment = experiment_class(
        output_dir=output_dir,
        test_mode=True  # Set to True to bypass notes requirement initially
    )
    
    # Initialize data manager with explicit data directory
    data_dir = Path(__file__).parent.parent / 'data'
    data_manager = DataManager(data_dir)
    
    try:
        data_manager.load_data()
        
        # Stratified sampling based on NumberofDependents
        train_sample = stratified_sample(
            data_manager.finaltrain, 
            'NumberofDependents', 
            sample_size
        )
        val_sample = stratified_sample(
            data_manager.finalvalidate, 
            'NumberofDependents', 
            sample_size//2
        )
        test_sample = stratified_sample(
            data_manager.finaltest, 
            'NumberofDependents', 
            sample_size//2
        )
        
        print(f"\nRunning {name} experiment with {len(train_sample)} training samples...")
        print("Sample distribution:")
        print_distribution(train_sample, 'NumberofDependents')
        
        start_time = datetime.now()
        
        experiment.run(
            train_data=train_sample,
            val_data=val_sample,
            test_data=test_sample
        )
        
        # Get and save metadata with notes
        metadata = experiment.get_experiment_metadata()
        metadata.update({
            'runtime': (datetime.now() - start_time).total_seconds(),
            'sample_sizes': {
                'train': len(train_sample),
                'val': len(val_sample),
                'test': len(test_sample)
            },
            'experiment_notes': notes,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save metadata with timestamp in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_path = output_dir / f'experiment_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure data files exist in:", data_dir)
        sys.exit(1)

def stratified_sample(df: pd.DataFrame, strat_col: str, n: int) -> pd.DataFrame:
    """Take a stratified sample to maintain class distribution"""
    grouped = df.groupby(strat_col)
    sizes = (grouped.size() * (n / len(df))).astype(int)
    samples = []
    
    for name, group in grouped:
        sample_size = sizes[name]
        if sample_size > 0:
            samples.append(group.sample(n=sample_size, random_state=42))
    
    return pd.concat(samples).sample(frac=1, random_state=42)

def print_distribution(df: pd.DataFrame, column: str) -> None:
    """Print value distribution for a column"""
    dist = df[column].value_counts().sort_index()
    total = len(df)
    
    print(f"\n{column} distribution:")
    for val, count in dist.items():
        pct = count/total * 100
        print(f"{val}: {count} ({pct:.1f}%)")

def compare_experiments(nn_metadata: dict, xgb_metadata: dict, svr_metadata: dict):
    """Compare and print experiment results"""
    print("\nExperiment Comparison:")
    print("-" * 50)
    
    # Compare hardware usage
    print("\nHardware Configuration:")
    for name, meta in [
        ("Neural Network", nn_metadata),
        ("XGBoost", xgb_metadata), 
        ("SVR", svr_metadata)
    ]:
        hw = meta['hardware_config']
        print(f"\n{name}:")
        print(f"GPU Enabled: {hw['gpu_enabled']}")
        if hw['device_name']:
            print(f"Device: {hw['device_name']}")
            print(f"Memory Used: {hw['memory_allocated']:.2f} MB")
    
    # Compare performance metrics
    print("\nPerformance Metrics:")
    for name, meta in [
        ("Neural Network", nn_metadata),
        ("XGBoost", xgb_metadata), 
        ("SVR", svr_metadata)
    ]:
        print(f"\n{name} Validation RMSLE: {meta.get('val_rmsle', 'N/A'):.4f}")
        print(f"{name} Test RMSLE: {meta.get('test_rmsle', 'N/A'):.4f}")
    
    # Print runtime comparison
    print("\nRuntime Comparison:")
    print(f"Neural Network: {nn_metadata['runtime']:.2f} seconds")
    print(f"XGBoost: {xgb_metadata['runtime']:.2f} seconds")
    print(f"SVR: {svr_metadata['runtime']:.2f} seconds")

def get_experiment_notes() -> str:
    """Interactively collect experiment notes with proper structure"""
    print("\nEnter experiment notes (press Enter to skip section, Ctrl+D to finish):")
    print("(Enter 's' to skip all remaining sections)")
    
    sections = {
        'hypothesis': 'What are you testing?',
        'methodology': 'How are you testing it?',
        'observations': 'Initial thoughts on results?',
        'next_steps': 'What should be tried next?'
    }
    
    notes = []
    for section, prompt in sections.items():
        print(f"\n{section}:")
        print(prompt)
        lines = []
        
        # Check if user wants to skip remaining sections
        try:
            first_input = input('> ')
            if first_input.lower() == 's':
                break
            elif first_input:  # Only add if not empty
                lines.append(f"- {first_input}")
                
            # Continue collecting lines for this section
            while True:
                try:
                    line = input('> ')
                    if not line:
                        break
                    lines.append(f"- {line}")
                except EOFError:
                    break
                    
            if lines:  # Only add section if it has content
                notes.append(f"{section}:\n" + "\n".join(lines))
                
        except EOFError:
            break
    
    return "\n\n".join(notes) if notes else "Quick run - no notes provided"

def main():
    """Run experiments with notes collection at the end"""
    # Run experiments
    nn_metadata = run_experiment(
        NeuralNetSegmentExperiment,
        'neuralnet_segment',
        sample_size=96000,  # Same sample size as other experiments
        notes="Pending user notes"
    )
    
    # Optional: Run other experiments
    xgb_metadata = run_experiment(
        XGBoostSegmentExperiment, 
        'xgboost_segment',
        sample_size=96000,
        notes="Pending user notes"
    )
    
    svr_metadata = run_experiment(
        SVRSegmentExperiment, 
        'svr_segment',
        sample_size=96000,
        notes="Pending user notes"
    )
    
    # Update compare_experiments function call to include neural net
    compare_experiments(nn_metadata, xgb_metadata, svr_metadata)
    
    # Collect notes for all experiments
    print("\nExperiments completed. Adding notes for documentation...")
    
    print("\nNeural Network Experiment Notes:")
    nn_notes = get_experiment_notes()
    
    print("\nXGBoost Experiment Notes:")
    xgb_notes = get_experiment_notes()
    
    print("\nSVR Experiment Notes:")
    svr_notes = get_experiment_notes()
    
    # Update metadata files with notes
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for name, metadata, notes in [
        ('neuralnet_segment', nn_metadata, nn_notes),
        ('xgboost_segment', xgb_metadata, xgb_notes),
        ('svr_segment', svr_metadata, svr_notes)
    ]:
        metadata['experiment_notes'] = notes
        output_dir = Path(__file__).parent / name
        metadata_path = output_dir / f'experiment_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()