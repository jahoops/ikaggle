import tensorflow as tf
import os
import datetime

def setup_tensorboard(log_dir):
    """Setup TensorBoard writer with directory cleanup"""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, 'tensorboard', current_time)
    return tf.summary.create_file_writer(log_dir)

def log_evolution_stats(writer, gen, min_fitness, avg_fitness, feature_count, sample_size, stagnant_gens):
    """Log evolution statistics to TensorBoard with better organization"""
    with writer.as_default():
        # Fitness metrics
        tf.summary.scalar('fitness/min', min_fitness, step=gen)
        tf.summary.scalar('fitness/avg', avg_fitness, step=gen)
        
        # Feature counts (now handling both dict and float inputs)
        if isinstance(feature_count, dict):
            # Original behavior for dictionary input
            tf.summary.scalar('features/base', feature_count.get('base', 0), step=gen)
            tf.summary.scalar('features/pairs', feature_count.get('pairs', 0), step=gen)
            tf.summary.scalar('features/triplets', feature_count.get('triplets', 0), step=gen)
            total_features = sum(feature_count.values())
        else:
            # New behavior for float input (total feature count)
            tf.summary.scalar('features/total', feature_count, step=gen)
            total_features = feature_count
        
        # Training metrics
        tf.summary.scalar('training/sample_size', sample_size, step=gen)
        tf.summary.scalar('training/stagnant_generations', stagnant_gens, step=gen)
        tf.summary.scalar('training/total_features', total_features, step=gen)
