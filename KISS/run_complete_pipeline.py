import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.data_processor import InsuranceDataProcessor
from utils.segmentation import InsuranceSegmenter
from model.group_trainer import GroupSpecialistTrainer
from model.super_nn import SuperNN
from model.diverse_models import DiverseModelFactory, ModelTrainer
import glob
from utils.feature_analyzer import combine_feature_importance_methods
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("Starting comprehensive insurance premium prediction pipeline...")
    
    # Create directories
    base_results_dir = '/home/USER/Projects/ikaggle/KISS/results'
    results_dir = f"{base_results_dir}/complete_pipeline"
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
    
    # Save processed data for future use
    processed_data_path = f"{results_dir}/processed_data"
    os.makedirs(processed_data_path, exist_ok=True)
    X_train.to_csv(f"{processed_data_path}/X_train.csv", index=False)
    X_test.to_csv(f"{processed_data_path}/X_test.csv", index=False)
    pd.DataFrame({'PremiumAmount': y_train}).to_csv(f"{processed_data_path}/y_train.csv", index=False)
    
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
        methods=['random_forest', 'correlation'],
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
    
    # Step 3: Create Feature Subsets for Different Model Types
    print("\n" + "="*80)
    print("STEP 3: CREATING FEATURE SUBSETS")
    print("="*80)
    
    # Create feature subsets
    feature_subsets = []
    
    # Subset 1: All features
    all_features = list(X_train_df.columns)
    feature_subsets.append(all_features)
    print(f"Subset 1: All features ({len(all_features)} features)")
    
    # Subset 2: Top features from importance analysis
    feature_subsets.append(top_features)
    print(f"Subset 2: Top features ({len(top_features)} features)")
    
    # Subset 3: Segment-specific important features
    segment_features = []
    for feature in top_features[:10]:  # Top 10 features from importance analysis
        segment_features.append(feature)
    # Add cluster identifier as feature
    X_train_df['Segment'] = groups
    X_val_df['Segment'] = segmenter.assign_segment(X_val_df)
    segment_features.append('Segment')
    feature_subsets.append(segment_features)
    print(f"Subset 3: Segment-specific features ({len(segment_features)} features)")
    
    # Save feature subsets for future use
    with open(f"{results_dir}/feature_subsets.txt", 'w') as f:
        for i, subset in enumerate(feature_subsets):
            f.write(f"Subset {i+1}: {','.join(subset)}\n")
    
    # Step 4: Train Base Models (if needed)
    print("\n" + "="*80)
    print("STEP 4: TRAINING BASE MODELS")
    print("="*80)
    
    # Check for existing submission files
    submission_files = glob.glob(f"{base_results_dir}/submission_model_*.csv")
    
    # If no submission files exist, train basic models
    if not submission_files:
        print("\nNo existing model submissions found. Training new models...")
        
        # Train diverse models using our processed data
        model_trainer = ModelTrainer(results_dir=f"{results_dir}/base_models")
        
        # Use diverse model architectures to ensure we cover the data well
        training_results = model_trainer.train_diverse_models(
            X=X_train_df,
            y=y_train,
            feature_subsets=feature_subsets,
            test_data=X_test,
            n_folds=3  # Use fewer folds for speed
        )
        
        # Get the submissions we just created
        submission_files = glob.glob(f"{results_dir}/base_models/*_submission.csv")
        print(f"Created {len(submission_files)} new model submissions")
    else:
        print(f"\nFound {len(submission_files)} existing model submissions")
    
    # Step 5: Evaluate Models by Customer Segment
    print("\n" + "="*80)
    print("STEP 5: EVALUATING MODELS BY CUSTOMER SEGMENT")
    print("="*80)
    
    # Use the group trainer to evaluate models by segment
    group_trainer = GroupSpecialistTrainer(results_dir=f"{results_dir}/group_models")
    
    # Assign groups to validation data
    val_groups = X_val_df['Segment'].values if 'Segment' in X_val_df.columns else segmenter.assign_segment(X_val_df)
    
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
    
    # Step 6: Train Specialized Models for Each Segment
    print("\n" + "="*80)
    print("STEP 6: TRAINING SPECIALIZED MODELS FOR EACH SEGMENT")
    print("="*80)
    
    # Try to load existing models or use basic architecture if needed
    general_models = []
    
    try:
        # First try to load pre-trained models
        for i in range(3):  # Try loading first 3 models
            try:
                model = SuperNN.load_model(i)
                general_models.append(model)
                print(f"Loaded pre-existing model {i}")
            except Exception as e:
                print(f"Could not load model {i}: {e}")
        
        if not general_models:
            raise ValueError("No existing models could be loaded")
            
    except Exception:
        print("Creating simple models as fallback...")
        
        # Create simple models for each subset
        for i, features_subset in enumerate(feature_subsets):
            # Drop 'Segment' feature if present for model input
            if 'Segment' in features_subset:
                features_to_use = [f for f in features_subset if f != 'Segment']
            else:
                features_to_use = features_subset
                
            # Create model using simple architecture
            X_subset = X_train_df[features_to_use]
            scaler = StandardScaler()
            X_subset_scaled = scaler.fit_transform(X_subset)
            
            # Use simple model factory to create model
            model, _ = DiverseModelFactory.create_model('wide_and_deep', input_dim=X_subset_scaled.shape[1])
            
            # Create simplified wrapper compatible with our API
            class SimpleModelWrapper:
                def __init__(self, keras_model, input_shape):
                    self.model = keras_model
                    self.input_dim = input_shape[1]
                    self.num_combinations = 1  # Default
                    self.model_id = f"fallback_model_{i}"
                    self.architecture_id = 0  # Default
                    self.feature_combinations = []  # Default
                
                def predict(self, X):
                    # Handle different input formats
                    if isinstance(X, list):
                        return self.model.predict(X[0])
                    return self.model.predict(X)
                    
            # Create wrapper
            wrapper = SimpleModelWrapper(model, X_subset_scaled.shape)
            general_models.append(wrapper)
    
    if not general_models:
        print("\nCould not create any models. Exiting pipeline.")
        return
    
    print(f"\nPrepared {len(general_models)} models for specialization.")
    
    # Remove segment column from training/validation data for model input
    if 'Segment' in X_train_df.columns:
        X_train_df = X_train_df.drop('Segment', axis=1)
    if 'Segment' in X_val_df.columns:
        X_val_df = X_val_df.drop('Segment', axis=1)
    
    # Prepare data for model training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test)
    
    # X_train needs to be a list for our models with [original_features, combined_features]
    # For simplicity, we'll use all features as original and an empty array for combined
    train_input = [X_train_scaled, np.zeros((X_train_scaled.shape[0], 1))]
    val_input = [X_val_scaled, np.zeros((X_val_scaled.shape[0], 1))]
    test_input = [X_test_scaled, np.zeros((X_test_scaled.shape[0], 1))]
    
    # Assign groups to training data (recompute since we dropped the segment column)
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
    
    # Step 7: Create Group-Based Ensemble
    print("\n" + "="*80)
    print("STEP 7: CREATING GROUP-BASED ENSEMBLE")
    print("="*80)
    
    # Create ensemble that uses the best specialized model for each customer segment
    ensemble_path = group_trainer.create_group_based_ensemble(
        specialized_models,
        segmenter,
        test_input
    )
    
    print(f"\nGroup-based ensemble prediction saved to: {ensemble_path}")
    
    # Copy final submission to base results directory for easy access
    import shutil
    final_path = f"{base_results_dir}/final_group_ensemble_submission.csv"
    shutil.copy(ensemble_path, final_path)
    print(f"\nFinal submission copied to: {final_path}")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
