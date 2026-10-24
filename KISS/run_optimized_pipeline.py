import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import glob
from utils.data_processor import InsuranceDataProcessor
from utils.segmentation import InsuranceSegmenter
from model.group_trainer import GroupSpecialistTrainer
from model.super_nn import SuperNN
from model.diverse_models import DiverseModelFactory, ModelTrainer
from utils.feature_analyzer import combine_feature_importance_methods
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    # Start timing the pipeline
    start_time = time.time()
    
    print("Starting optimized insurance premium prediction pipeline...")
    
    # Create directories
    base_results_dir = '/home/USER/Projects/ikaggle/KISS/results'
    results_dir = f"{base_results_dir}/complete_pipeline"
    os.makedirs(results_dir, exist_ok=True)
    
    processed_data_path = f"{results_dir}/processed_data"
    segmentation_path = f"{results_dir}/segmentation"
    base_models_path = f"{results_dir}/base_models"
    group_models_path = f"{results_dir}/group_models"
    
    # Step 1: Data Processing - Check if we can skip
    print("\n" + "="*80)
    print("STEP 1: DATA PROCESSING")
    print("="*80)
    
    X_train_file = f"{processed_data_path}/X_train.csv"
    X_test_file = f"{processed_data_path}/X_test.csv"
    y_train_file = f"{processed_data_path}/y_train.csv"
    
    if os.path.exists(X_train_file) and os.path.exists(X_test_file) and os.path.exists(y_train_file):
        print("Found processed data files. Loading instead of reprocessing...")
        X_train = pd.read_csv(X_train_file)
        X_test = pd.read_csv(X_test_file)
        y_train = pd.read_csv(y_train_file)['PremiumAmount'].values
        print(f"Loaded processed data: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}")
    else:
        print("Processed data not found. Running full data processing...")
        # Define file paths
        train_path = '/home/USER/Projects/ikaggle/KISS/data/train_raw.csv'
        test_path = '/home/USER/Projects/ikaggle/KISS/data/contest_raw.csv'
        
        processor = InsuranceDataProcessor(
            train_path=train_path,
            test_path=test_path,
            results_dir=f"{results_dir}/data_processing"
        )
        
        processor.prepare_data()
        X_train, X_test, y_train = processor.get_processed_data()
        
        # Save processed data for future use
        os.makedirs(processed_data_path, exist_ok=True)
        X_train.to_csv(X_train_file, index=False)
        X_test.to_csv(X_test_file, index=False)
        pd.DataFrame({'PremiumAmount': y_train}).to_csv(y_train_file, index=False)
        
        print("Data processing complete and results saved.")
    
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
    
    feature_importance_file = f"{segmentation_path}/feature_importance_combined.png"
    cluster_files = glob.glob(f"{segmentation_path}/cluster_assignments_kmeans_*.csv")
    segment_stats_file = f"{segmentation_path}/segment_statistics.csv"
    top_features_file = f"{segmentation_path}/top_features.txt"
    
    if (os.path.exists(feature_importance_file) and cluster_files and 
        os.path.exists(segment_stats_file) and os.path.exists(top_features_file)):
        print("Found segmentation results. Loading instead of recomputing...")
        
        # Load top features
        with open(top_features_file, 'r') as f:
            top_features = [line.strip() for line in f.readlines()]
        
        # Load cluster assignments
        cluster_df = pd.read_csv(cluster_files[0])
        groups = cluster_df['Cluster'].values
        
        # Load segment statistics
        segment_stats = pd.read_csv(segment_stats_file)
        
        # Create segmenter but don't recompute
        segmenter = InsuranceSegmenter(results_dir=segmentation_path)
        segmenter.clusters = groups  # Set clusters directly
        segmenter.top_features = top_features  # Set top features directly
        
        # FIXED: Initialize scalers and simple models for prediction
        try:
            # Load training data for initializing the model - if available
            train_X_file = f"{segmentation_path}/train_features_for_segments.csv"
            if os.path.exists(train_X_file):
                print("Found training data for segments. Initializing models...")
                train_X = pd.read_csv(train_X_file)
                
                # Filter to available top features
                avail_features = [f for f in top_features if f in train_X.columns]
                train_X = train_X[avail_features]
                
                # Initialize scaler with training data
                segmenter.scaler = StandardScaler()
                segmenter.scaler.fit(train_X)
                
                # Initialize a simple clustering model (KMeans)
                from sklearn.cluster import KMeans
                segmenter.cluster_model = KMeans(n_clusters=len(np.unique(groups)), random_state=42, n_init=10)
                segmenter.cluster_model.fit(segmenter.scaler.transform(train_X))
                print("Successfully initialized segment prediction models")
            else:
                print("No training features found. Will use simpler fallback in assign_segment()")
        except Exception as e:
            print(f"Error initializing segmentation models: {e}")
            print("Will use fallbacks in assign_segment()")
    
    else:
        print("Segmentation results not found. Running full segmentation...")
        segmenter = InsuranceSegmenter(results_dir=segmentation_path)
        
        # Use both methods and combine results
        top_features = combine_feature_importance_methods(
            segmenter,
            X_train_df, 
            y_train_log_split,
            methods=['random_forest', 'correlation'],
            n_features=15
        )
        
        # Save top features for future use
        os.makedirs(segmentation_path, exist_ok=True)
        with open(top_features_file, 'w') as f:
            for feature in top_features:
                f.write(f"{feature}\n")
        
        print(f"Top features: {top_features}")
        
        # Create customer segments using combined top features
        optimal_clusters = 4
        groups = segmenter.create_segments(X_train_df, y_train_log_split, method='kmeans', n_clusters=optimal_clusters)
        
        # Analyze segments
        segment_stats = segmenter.analyze_segments(X_train_df, y_train)
        
        # FIXED: Save training data for future segment assignment
        X_features = X_train_df[top_features].copy()
        X_features.to_csv(f"{segmentation_path}/train_features_for_segments.csv", index=False)
    
    # Print segment insights
    print("\nCustomer Segment Insights:")
    for cluster_id, stats in segment_stats.iterrows():
        print(f"\nSegment {stats['Cluster']} ({stats['Size']} customers, {stats['Percentage']:.1f}% of data):")
        print(f"  Premium Mean: ${stats['Target_Mean']:.2f}")
        print(f"  Premium Range: ${stats['Target_Min']:.2f} - ${stats['Target_Max']:.2f}")
    
    # Step 3: Create Feature Subsets
    print("\n" + "="*80)
    print("STEP 3: CREATING FEATURE SUBSETS")
    print("="*80)
    
    feature_subsets_file = f"{results_dir}/feature_subsets.txt"
    
    if os.path.exists(feature_subsets_file):
        print("Found feature subsets. Loading instead of recreating...")
        feature_subsets = []
        current_subset = []
        
        with open(feature_subsets_file, 'r') as f:
            for line in f:
                if line.startswith("Subset "):
                    if current_subset:
                        feature_subsets.append(current_subset)
                        current_subset = []
                    # Extract features from line
                    features_part = line.split(":", 1)[1]
                    current_subset = features_part.strip().split(',')
        
        if current_subset:  # Add the last subset
            feature_subsets.append(current_subset)
            
        print(f"Loaded {len(feature_subsets)} feature subsets")
    else:
        print("Feature subsets not found. Creating feature subsets...")
        
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
        
        # Add segment information
        X_train_df['Segment'] = groups
        X_val_df['Segment'] = segmenter.assign_segment(X_val_df)
        segment_features.append('Segment')
        feature_subsets.append(segment_features)
        print(f"Subset 3: Segment-specific features ({len(segment_features)} features)")
        print("Note: 'Segment' will be automatically filtered out when processing test data")
        
        # Save feature subsets for future use
        with open(feature_subsets_file, 'w') as f:
            for i, subset in enumerate(feature_subsets):
                f.write(f"Subset {i+1}: {','.join(subset)}\n")
    
    # Step 4: Train Base Models (if needed)
    print("\n" + "="*80)
    print("STEP 4: TRAINING BASE MODELS")
    print("="*80)
    
    submission_files = glob.glob(f"{base_results_dir}/submission_model_*.csv")
    pipeline_submission_files = glob.glob(f"{base_models_path}/*_submission.csv")
    
    all_submission_files = submission_files + pipeline_submission_files
    
    if all_submission_files:
        print(f"Found {len(all_submission_files)} existing model submissions. Skipping base model training...")
    else:
        print("No existing model submissions found. Training new models...")
        
        # Train diverse models using our processed data
        model_trainer = ModelTrainer(results_dir=base_models_path)
        
        training_results = model_trainer.train_diverse_models(
            X=X_train_df,
            y=y_train,
            feature_subsets=feature_subsets,
            test_data=X_test,
            n_folds=3  # Use fewer folds for speed
        )
        
        # Get the submissions we just created
        all_submission_files = glob.glob(f"{base_models_path}/*_submission.csv")
        print(f"Created {len(all_submission_files)} new model submissions")
    
    # Step 5: Evaluate Models by Customer Segment
    print("\n" + "="*80)
    print("STEP 5: EVALUATING MODELS BY CUSTOMER SEGMENT")
    print("="*80)
    
    performance_file = f"{group_models_path}/segment_performance.csv"
    
    if os.path.exists(performance_file):
        print("Found segment performance data. Loading instead of reevaluating...")
        performance_df = pd.read_csv(performance_file)
        
        # Recreate performance dictionary
        performance = {}
        for _, row in performance_df.iterrows():
            model_id = row['Model']
            group_id = int(row['Group'])
            score = row['RMSLE']
            
            if model_id not in performance:
                performance[model_id] = {}
            
            performance[model_id][group_id] = score
            
        # Initialize group trainer
        group_trainer = GroupSpecialistTrainer(results_dir=group_models_path)
        
        # Get val_groups
        if 'Segment' in X_val_df.columns:
            val_groups = X_val_df['Segment'].values
        else:
            val_groups = segmenter.assign_segment(X_val_df)
            
        # Set up group assignments
        group_trainer.group_assignments = {'test': val_groups}
        
        # Initialize group_submission_files in trainer
        unique_groups = np.unique(val_groups)
        group_trainer.group_submission_files = {group_id: [] for group_id in unique_groups}
        group_trainer.performance = performance
        
        # Find best models per group
        group_trainer._find_best_models_per_group()
    else:
        print("Segment performance data not found. Evaluating models by segment...")
        
        # Use the group trainer to evaluate models by segment
        group_trainer = GroupSpecialistTrainer(results_dir=group_models_path)
        
        # Assign groups to validation data
        if 'Segment' in X_val_df.columns:
            val_groups = X_val_df['Segment'].values
        else:
            val_groups = segmenter.assign_segment(X_val_df)
            
        # Assign groups and evaluate by segment
        group_assignments = {'test': val_groups}
        group_trainer.group_assignments = group_assignments
        
        # Evaluate each model's performance on each segment
        performance = group_trainer.evaluate_models_by_group(
            all_submission_files,
            val_groups,
            y_val
        )
        
        # Save performance data for future use
        performance_data = []
        for model_id, group_scores in performance.items():
            for group_id, score in group_scores.items():
                performance_data.append({
                    'Model': model_id,
                    'Group': group_id,
                    'RMSLE': score
                })
                
        pd.DataFrame(performance_data).to_csv(performance_file, index=False)
    
    # Visualize segment performance
    performance_viz_file = f"{results_dir}/model_segment_performance.png"
    if not os.path.exists(performance_viz_file):
        print("Generating segment performance visualization...")
        segment_perf_data = []
        for model_id, group_scores in group_trainer.performance.items():
            for group_id, score in group_scores.items():
                segment_perf_data.append({
                    'Model': model_id,
                    'Segment': f"Segment {group_id}",
                    'RMSLE': score
                })
                
        segment_perf_df = pd.DataFrame(segment_perf_data)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        heatmap_data = segment_perf_df.pivot(index='Model', columns='Segment', values='RMSLE')
        import seaborn as sns
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu_r', fmt='.4f')
        plt.title('Model Performance (RMSLE) by Customer Segment')
        plt.tight_layout()
        plt.savefig(performance_viz_file)
        print("Performance visualization saved.")
    
    # Step 6: Train Specialized Models for Each Segment
    print("\n" + "="*80)
    print("STEP 6: TRAINING SPECIALIZED MODELS FOR EACH SEGMENT")
    print("="*80)
    
    specialized_models_files = glob.glob(f"{group_models_path}/specialized_model_*_group*.csv")
    unique_groups = np.unique(group_trainer.group_assignments['test'])
    
    # Check if we have specialized models for each group
    all_groups_covered = all(
        any(f"_group{group_id}.csv" in f for f in specialized_models_files)
        for group_id in unique_groups
    )
    
    if specialized_models_files and all_groups_covered:
        print(f"Found {len(specialized_models_files)} specialized model files covering all segments. Skipping specialized model training...")
        # We'll use these files directly in the ensemble step
    else:
        print("Missing specialized models for some segments. Training specialized models...")
        
        # Try to load existing general models or use basic architecture if needed
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
                X_subset = X_train_df[[f for f in features_to_use if f in X_train_df.columns]]
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
        X_train_no_segment = X_train_df.drop('Segment', axis=1) if 'Segment' in X_train_df.columns else X_train_df
        X_val_no_segment = X_val_df.drop('Segment', axis=1) if 'Segment' in X_val_df.columns else X_val_df
        
        # Prepare data for model training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_no_segment)
        X_val_scaled = scaler.transform(X_val_no_segment)
        X_test_scaled = scaler.transform(X_test)
        
        # X_train needs to be a list for our models with [original_features, combined_features]
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
    
    # Step 7: Create Group-Based Ensemble
    print("\n" + "="*80)
    print("STEP 7: CREATING GROUP-BASED ENSEMBLE")
    print("="*80)
    
    final_ensemble_path = f"{base_results_dir}/final_group_ensemble_submission.csv"
    group_ensemble_path = f"{group_models_path}/group_ensemble_submission.csv"
    
    if os.path.exists(final_ensemble_path) and os.path.getmtime(final_ensemble_path) > os.path.getmtime(__file__):
        print("Found up-to-date final ensemble. Skipping ensemble creation...")
        print(f"Final submission available at: {final_ensemble_path}")
    else:
        print("Creating group-based ensemble...")
        
        # Check if we just trained specialized models in this run
        if 'specialized_models' in locals():
            print("Using newly trained specialized models for ensemble...")
            # Use the specialized models directly
            ensemble_path = group_trainer.create_group_based_ensemble(
                specialized_models,
                segmenter,
                test_input
            )
            
            # Copy to final path
            import shutil
            shutil.copy(ensemble_path, final_ensemble_path)
            print(f"Final ensemble saved to: {final_ensemble_path}")
        # If we have existing specialized models, use direct submission files
        elif specialized_models_files and all_groups_covered:
            # We need to 1) get the list of files per group, 2) create dummy specialized models for ensemble
            from collections import defaultdict
            group_files = defaultdict(list)
            
            # Group files by segment
            for file in specialized_models_files:
                # Extract group ID from filename
                import re
                match = re.search(r'_group(\d+).csv', file)
                if match:
                    group_id = int(match.group(1))
                    group_files[group_id].append(file)
            
            # Create an ensemble directly from these files
            import shutil
            
            # Use the existing ensemble function from group_trainer
            if os.path.exists(group_ensemble_path):
                print(f"Found group ensemble at {group_ensemble_path}. Copying to final location...")
                shutil.copy(group_ensemble_path, final_ensemble_path)
            else:
                print("No existing group ensemble found. Creating from submission files...")
                
                # Manual ensemble using submission files
                from model.ensemble import ensemble_submissions
                
                # Collect all specialized model submission files
                all_specialized_submissions = []
                for group_id, files in group_files.items():
                    all_specialized_submissions.extend(files)
                
                # Create ensemble from all specialized submissions
                if all_specialized_submissions:
                    # MODIFIED: Explicitly use robust ensemble method
                    print("Using robust ensemble method for specialized submissions...")
                    ensemble_path = ensemble_submissions(
                        all_specialized_submissions, 
                        output_path=final_ensemble_path,
                        method='median'  # Changed from 'weighted' to 'median' for robustness
                    )
                    print(f"Created ensemble from {len(all_specialized_submissions)} specialized submissions")
                elif all_submission_files:
                    print("No specialized submissions found. Using base model submissions...")
                    ensemble_path = ensemble_submissions(
                        all_submission_files[:5],  # Use top 5 submissions
                        output_path=final_ensemble_path,
                        method='weighted'
                    )
                else:
                    print("No submission files found. Could not create ensemble.")
        else:
            # If we've reached here, we need to create an emergency submission
            print("No specialized models found. Creating emergency submission...")
            
            emergency_submission = pd.DataFrame({
                'id': range(1200000, 1200000 + len(X_test)),
                'Premium Amount': np.ones(len(X_test)) * 500.0  # Default value
            })
            
            emergency_submission.to_csv(final_ensemble_path, index=False)
            print(f"Created emergency submission at {final_ensemble_path}")
    
    # Calculate total runtime
    end_time = time.time()
    runtime_minutes = (end_time - start_time) / 60
    print(f"\nPipeline completed in {runtime_minutes:.1f} minutes.")

if __name__ == "__main__":
    main()
