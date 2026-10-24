import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.data_processor import InsuranceDataProcessor
from utils.segmentation import InsuranceSegmenter
from model.group_trainer import GroupSpecialistTrainer
from model.super_nn import SuperNN
import glob
from utils.feature_analyzer import combine_feature_importance_methods

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("Starting comprehensive group-based insurance premium prediction pipeline...")
    
    # Create directories
    results_dir = '/home/USER/Projects/ikaggle/KISS/results/group_pipeline'
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Data Processing with InsuranceDataProcessor
    print("\n" + "="*80)
    print("STEP 1: COMPREHENSIVE DATA PROCESSING")
    print("="*80)
    
    # Define file paths
    train_path = '/home/USER/Projects/ikaggle/KISS/data/train_raw.csv'
    test_path = '/home/USER/Projects/ikaggle/KISS/data/contest_raw.csv'
    
    # Initialize and run data processor
    processor = InsuranceDataProcessor(
        train_path=train_path,
        test_path=test_path,
        results_dir=f"{results_dir}/data_processing"
    )
    
    processor.prepare_data()
    X_train, X_test, y_train = processor.get_processed_data()
    
    # Create log-transformed target for modeling
    y_train_log = np.log1p(y_train)
    
    # Split data into train and validation sets
    from sklearn.model_selection import train_test_split
    X_train_df, X_val_df, y_train_log_split, y_val_log = train_test_split(
        X_train, y_train_log, test_size=0.2, random_state=42
    )
    y_val = np.expm1(y_val_log)  # Keep original scale for evaluation
    
    # Step 2: Feature Importance Analysis and Customer Segmentation
    print("\n" + "="*80)
    print("STEP 2: FEATURE IMPORTANCE & CUSTOMER SEGMENTATION")
    print("="*80)
    
    segmenter = InsuranceSegmenter(results_dir=f"{results_dir}/segmentation")
    
    # Use both methods and combine results
    print("\nAnalyzing feature importance to identify key factors:")
    top_features = combine_feature_importance_methods(
        segmenter,
        X_train_df, 
        y_train_log_split,
        methods=['random_forest', 'correlation', 'gradient_boosting'],
        n_features=15
    )
    
    print(f"\nTop features across all methods: {top_features}")
    
    # Create customer segments using combined top features
    print("\nCreating customer segments for specialized model training:")
    optimal_clusters = 4  # You could dynamically determine this
    groups = segmenter.create_segments(X_train_df, y_train_log_split, method='kmeans', n_clusters=optimal_clusters)
    
    # Analyze the segments to understand characteristics
    segment_stats = segmenter.analyze_segments(X_train_df, y_train)
    
    # Print segment insights
    print("\nCustomer Segment Insights:")
    for cluster_id, stats in segment_stats.iterrows():
        print(f"\nSegment {stats['Cluster']} ({stats['Size']} customers, {stats['Percentage']:.1f}% of data):")
        print(f"  Premium Mean: ${stats['Target_Mean']:.2f}")
        print(f"  Premium Range: ${stats['Target_Min']:.2f} - ${stats['Target_Max']:.2f}")
        
        # Print top 3 distinctive features
        features_to_report = [col.replace('_Mean', '') for col in stats.index if col.endswith('_Mean')][:3]
        for feature in features_to_report:
            if f'{feature}_Mean' in stats:
                print(f"  {feature}: {stats[f'{feature}_Mean']:.2f}")
    
    # Step 3: Evaluate Existing Models by Customer Segment
    print("\n" + "="*80)
    print("STEP 3: EVALUATING EXISTING MODELS BY SEGMENT")
    print("="*80)
    
    # Use the group trainer to evaluate models by segment
    group_trainer = GroupSpecialistTrainer(results_dir=f"{results_dir}/group_models")
    
    # Assign groups to validation data
    val_groups = segmenter.assign_segment(X_val_df)
    
    # Find existing submission files from previous model runs
    submission_files = glob.glob('/home/USER/Projects/ikaggle/KISS/results/submission_model_*.csv')
    
    if not submission_files:
        print("\nNo existing model submissions found. Please run regular ensemble training first.")
        return
    
    print(f"\nFound {len(submission_files)} model submission files to evaluate by segment.")
    
    # Assign groups and evaluate by segment
    group_assignments = {'test': val_groups}
    group_trainer.group_assignments = group_assignments
    
    # Evaluate each model's performance on each segment
    performance = group_trainer.evaluate_models_by_group(
        submission_files,
        val_groups,
        y_val
    )
    
    # Visualize segment performance
    if performance:
        segment_perf_data = []
        for model_id, group_scores in performance.items():
            for group_id, score in group_scores.items():
                segment_perf_data.append({
                    'Model': model_id,
                    'Segment': f"Segment {group_id}",
                    'RMSLE': score
                })
                
        segment_perf_df = pd.DataFrame(segment_perf_data)
        
        # Plot heatmap of model performance by segment
        plt.figure(figsize=(12, 8))
        heatmap_data = segment_perf_df.pivot(index='Model', columns='Segment', values='RMSLE')
        import seaborn as sns
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu_r', fmt='.4f')
        plt.title('Model Performance (RMSLE) by Customer Segment')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/model_segment_performance.png")
        
        print("\nModel performance by segment visualization saved.")
    
    # Step 4: Train Specialized Models for Each Segment
    print("\n" + "="*80)
    print("STEP 4: TRAINING SPECIALIZED MODELS FOR EACH SEGMENT")
    print("="*80)
    
    # Load existing general models
    general_models = []
    for i in range(7):  # Try to load all 7 standard models
        try:
            model = SuperNN.load_model(i)
            general_models.append(model)
            print(f"Loaded model {i}")
        except Exception as e:
            print(f"Could not load model {i}: {e}")
    
    if not general_models:
        print("\nNo existing models found. Please train general models first.")
        return
    
    print(f"\nLoaded {len(general_models)} general models to specialize for each segment.")
    
    # Prepare data for model training
    from sklearn.preprocessing import StandardScaler
    
    # Use standardized features for neural networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test)
    
    # X_train needs to be a list for our models with [original_features, combined_features]
    # For simplicity, we'll use all features as original and an empty array for combined
    train_input = [X_train_scaled, np.zeros((X_train_scaled.shape[0], 1))]
    val_input = [X_val_scaled, np.zeros((X_val_scaled.shape[0], 1))]
    test_input = [X_test_scaled, np.zeros((X_test_scaled.shape[0], 1))]
    
    # Assign groups to training data
    train_groups = groups
    group_trainer.group_assignments = {'train': train_groups, 'test': val_groups}
    
    # Train specialized models for each segment
    specialized_models = group_trainer.train_specialized_models(
        general_models,
        train_input,
        y_train_log_split,
        val_input,
        y_val_log
    )
    
    # Step 5: Create Group-Based Ensemble
    print("\n" + "="*80)
    print("STEP 5: CREATING GROUP-BASED ENSEMBLE")
    print("="*80)
    
    # Create ensemble that uses the best specialized model for each customer segment
    ensemble_path = group_trainer.create_group_based_ensemble(
        specialized_models,
        segmenter,
        test_input
    )
    
    print(f"\nGroup-based ensemble prediction saved to: {ensemble_path}")
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
