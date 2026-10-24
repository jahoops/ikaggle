import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def ensemble_submissions(submission_files, output_path=None, weights=None, method='weighted'):
    """
    Ensemble multiple submission files without loading models
    
    Args:
        submission_files: List of paths to submission files
        output_path: Path to save the ensembled submission
        weights: Optional list of weights for each submission
        method: Ensembling method ('weighted', 'mean', 'median', 'rank', 'diversity')
    
    Returns:
        Path to the ensembled submission file
    """
    print(f"Ensembling {len(submission_files)} submission files...")
    submissions = []
    valid_submissions = []
    valid_weights = []
    
    # Load all submission files
    for i, file_path in enumerate(submission_files):
        try:
            if os.path.exists(file_path):
                sub = pd.read_csv(file_path)
                submissions.append(sub)
                valid_submissions.append(file_path)
                if weights is not None:
                    valid_weights.append(weights[i])
            else:
                print(f"Warning: Submission file not found: {file_path}")
        except Exception as e:
            print(f"Error loading submission file {file_path}: {e}")
    
    if not submissions:
        raise ValueError("No valid submission files found")
        
    print(f"Successfully loaded {len(submissions)} submission files")
    
    # Check if all submissions have the same structure
    ids = submissions[0]['id'].values
    for i, sub in enumerate(submissions):
        if not np.array_equal(sub['id'].values, ids):
            print(f"Warning: Submission {valid_submissions[i]} has different IDs")
    
    # Update weights if needed
    if weights is None or len(valid_weights) != len(submissions):
        valid_weights = [1.0] * len(submissions)
    
    # Normalize weights
    total_weight = sum(valid_weights)
    if total_weight > 0:
        valid_weights = [w / total_weight for w in valid_weights]
    else:
        valid_weights = [1.0 / len(submissions)] * len(submissions)
    
    # Create matrix of all predictions
    all_preds = np.zeros((len(ids), len(submissions)))
    for i, sub in enumerate(submissions):
        # Handle different column names
        pred_col = 'Premium Amount' if 'Premium Amount' in sub.columns else 'PremiumAmount'
        all_preds[:, i] = sub[pred_col].values
    
    # Fix input submissions BEFORE ensembling
    print("Pre-processing individual submission files...")
    for i in range(all_preds.shape[1]):
        negative_count = np.sum(all_preds[:, i] < 0)
        if negative_count > 0:
            print(f"  Fixing {negative_count} negative values in submission {i+1}")
            all_preds[:, i] = np.maximum(all_preds[:, i], 0.0)
            
            # Apply small minimum for very small values
            very_small = (all_preds[:, i] > 0) & (all_preds[:, i] < 0.01)
            very_small_count = np.sum(very_small)
            if very_small_count > 0:
                all_preds[:, i] = np.where(very_small, 0.5, all_preds[:, i])
    
    # Apply ensembling method
    if method == 'weighted':
        ensembled_preds = np.sum(all_preds * np.array(valid_weights), axis=1)
    elif method == 'mean':
        ensembled_preds = np.mean(all_preds, axis=1)
    elif method == 'median':
        ensembled_preds = np.median(all_preds, axis=1)
    elif method == 'rank':
        # Rank-based averaging (more robust to outliers)
        ranks = np.zeros_like(all_preds)
        for i in range(all_preds.shape[1]):
            ranks[:, i] = all_preds[:, i].argsort().argsort()
        ranks = ranks / all_preds.shape[0]  # Normalize ranks
        ensembled_preds = np.sum(ranks * np.array(valid_weights), axis=1)
        # Map back to the original scale using quantiles
        all_values = all_preds.flatten()
        quantiles = np.percentile(all_values, np.linspace(0, 100, 101))
        for i in range(len(ensembled_preds)):
            rank_pct = ensembled_preds[i] * 100
            bin_idx = min(int(rank_pct), 99)
            ensembled_preds[i] = quantiles[bin_idx] + (rank_pct - bin_idx) * (quantiles[bin_idx+1] - quantiles[bin_idx])
    elif method == 'diversity':
        # Calculate correlation matrix between predictions
        corr_matrix = np.zeros((len(submissions), len(submissions)))
        for i in range(len(submissions)):
            for j in range(len(submissions)):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = pearsonr(all_preds[:, i], all_preds[:, j])
                    corr_matrix[i, j] = corr
        
        # Calculate diversity scores (lower correlation = higher diversity)
        diversity_scores = 1 - np.mean(corr_matrix, axis=1)
        
        # Use diversity as weight
        diversity_weights = diversity_scores / np.sum(diversity_scores)
        ensembled_preds = np.sum(all_preds * diversity_weights.reshape(-1, 1).T, axis=1)
    else:
        raise ValueError(f"Unknown ensembling method: {method}")
    
    # Fix negative predictions after ensembling (redundant but important safeguard)
    if np.any(ensembled_preds < 0):
        negative_count = np.sum(ensembled_preds < 0)
        print(f"\nWarning: Found {negative_count} negative predictions ({negative_count/len(ensembled_preds)*100:.2f}%).")
        print("Setting negative values to zero and applying small positive minimum.")
        
        # First approach: Simply replace negatives with zero
        ensembled_preds = np.maximum(ensembled_preds, 0.0)
        
        # For values very close to zero, set a small positive minimum
        # (Insurance premiums are typically at least a few dollars)
        very_small_threshold = 0.01  # $0.01
        very_small_count = np.sum((ensembled_preds > 0) & (ensembled_preds < very_small_threshold))
        if very_small_count > 0:
            print(f"Found {very_small_count} predictions between 0 and {very_small_threshold}.")
            # Set floor of $0.50
            min_premium = 0.50
            ensembled_preds = np.where(
                (ensembled_preds > 0) & (ensembled_preds < very_small_threshold),
                min_premium,
                ensembled_preds
            )
    
    # For insurance premiums, restrict values to a reasonable range
    # Insurance premiums are typically > $100 and < $10,000
    unreasonable_count = np.sum(ensembled_preds > 10000)
    if unreasonable_count > 0:
        print(f"Capping {unreasonable_count} unusually high predictions (>$10,000)")
        ensembled_preds = np.minimum(ensembled_preds, 10000.0)
    
    # Create final submission DataFrame
    final_submission = pd.DataFrame({
        'id': ids,
        'Premium Amount': ensembled_preds
    })
    
    # Print statistics about the ensemble prediction
    print(f"\nEnsemble Prediction Statistics:")
    print(f"Method: {method}")
    print(f"Min: {np.min(ensembled_preds):.2f}")
    print(f"Max: {np.max(ensembled_preds):.2f}")
    print(f"Mean: {np.mean(ensembled_preds):.2f}")
    print(f"Std: {np.std(ensembled_preds):.2f}")
    print(f"25th percentile: {np.percentile(ensembled_preds, 25):.2f}")
    print(f"Median: {np.percentile(ensembled_preds, 50):.2f}")
    print(f"75th percentile: {np.percentile(ensembled_preds, 75)::.2f}")
    
    # Show correlation with individual models
    print("\nCorrelation with individual models:")
    for i, file in enumerate(valid_submissions):
        name = os.path.basename(file)
        corr = np.corrcoef(ensembled_preds, all_preds[:, i])[0, 1]
        print(f"  {name}: {corr:.4f}")
    
    # Save to file
    if output_path is None:
        output_path = '/home/USER/Projects/ikaggle/KISS/results/contest_submission.csv'
    
    final_submission.to_csv(output_path, index=False)
    print(f"Ensembled submission saved to {output_path}")
    
    return output_path

def get_optimal_weights(submission_files, validation_scores=None):
    """
    Calculate optimal weights based on validation scores and diversity
    
    Args:
        submission_files: List of submission file paths
        validation_scores: List of validation scores (lower is better)
        
    Returns:
        List of optimal weights
    """
    if validation_scores is None or len(validation_scores) != len(submission_files):
        print("No valid validation scores provided, using equal weights")
        return [1.0/len(submission_files)] * len(submission_files)
    
    # Load all submissions to calculate diversity
    submissions = []
    for file_path in submission_files:
        try:
            if os.path.exists(file_path):
                sub = pd.read_csv(file_path)
                submissions.append(sub)
            else:
                print(f"Warning: Submission file not found: {file_path}")
                # Add a placeholder to maintain indices
                submissions.append(None)
        except Exception as e:
            print(f"Error loading submission file {file_path}: {e}")
            submissions.append(None)
    
    # Calculate diversity scores between valid submissions
    diversity_scores = [0.0] * len(submissions)
    
    # Only calculate for valid submissions
    valid_indices = [i for i, sub in enumerate(submissions) if sub is not None]
    if len(valid_indices) > 1:
        # Create prediction matrix
        all_preds = np.zeros((len(submissions[valid_indices[0]]), len(valid_indices)))
        for i, idx in enumerate(valid_indices):
            pred_col = 'Premium Amount' if 'Premium Amount' in submissions[idx].columns else 'PremiumAmount'
            all_preds[:, i] = submissions[idx][pred_col].values
        
        # Calculate correlation matrix
        corr_matrix = np.zeros((len(valid_indices), len(valid_indices)))
        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = pearsonr(all_preds[:, i], all_preds[:, j])
                    corr_matrix[i, j] = corr
        
        # Calculate diversity as inverse of average correlation
        for i, idx in enumerate(valid_indices):
            mean_corr = (np.sum(corr_matrix[i, :]) - 1.0) / (len(valid_indices) - 1)
            diversity_scores[idx] = 1.0 - mean_corr
    
    # Combine validation scores and diversity
    final_scores = []
    for i, score in enumerate(validation_scores):
        if i in valid_indices:
            # Lower validation score is better, higher diversity is better
            # Normalize diversity to 0.1-0.3 range to avoid overwhelming validation scores
            diversity_impact = 0.1 + 0.2 * diversity_scores[i]
            # Inverse score (lower is better) and multiply by diversity factor
            final_scores.append((1.0 / (score + 1e-5)) * diversity_impact)
        else:
            final_scores.append(0.0)
    
    # Normalize to sum to 1
    total = sum(final_scores)
    if total > 0:
        weights = [score/total for score in final_scores]
    else:
        weights = [1.0/len(validation_scores)] * len(validation_scores)
    
    print("Optimal weights based on validation scores and diversity:")
    for i, (file, weight, score, div) in enumerate(zip(
        submission_files, weights, validation_scores, diversity_scores)):
        print(f"  Model {i}: {os.path.basename(file)} - Weight: {weight:.4f} "
              f"(Score: {score:.4f}, Diversity: {div:.2f})")
    
    return weights
