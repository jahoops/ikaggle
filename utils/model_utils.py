from pathlib import Path
from venv import logger
import joblib

def load_model(name: str, output_dir: Path):
    """Load model from versioned directory.

    Supports both scikit-learn and Keras models.
    """
    model_filename = f'best_{name}_model.pkl'
    model_path = output_dir / name / model_filename  # Adjusted path to include model directory
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if name in ['NeuralNet']:
        # Load Keras model
        model = tf.keras.models.load_model(str(model_path), custom_objects={'safe_rmsle': safe_rmsle}) # type: ignore
    else:
        # Load scikit-learn model
        model = joblib.load(str(model_path))
    
    logger.info(f"Loaded model '{name}' from {model_path}")
    return model

def save_model(model_info: dict, name: str, output_dir: Path):
    """Save model info to disk"""
    output_dir = Path(output_dir)
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the entire model_info dictionary
    model_path = model_dir / f'best_{name}_model.pkl'  # Ensure consistent naming
    try:
        joblib.dump(model_info, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model {name}: {str(e)}")
        raise

