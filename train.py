import argparse
from pathlib import Path
from experiments.models import MODEL_REGISTRY
from utils import config

def main():
    parser = argparse.ArgumentParser(description="Train ML models.")
    parser.add_argument('model_type', type=str, choices=MODEL_REGISTRY.keys(),
                        help='Type of model to train.')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with minimal settings.')
    args = parser.parse_args()
    model_type = args.model_type.lower()
    
    # Validate model type
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type '{model_type}'. Supported types: {list(MODEL_REGISTRY.keys())}")

    test_mode = args.test  # Capture the test mode flag
    
    # Instantiate the appropriate experiment class with test_mode
    ExperimentClass = MODEL_REGISTRY[model_type]
    experiment = ExperimentClass(
        output_dir=Path(config.get('experiment', 'output_path')) / model_type.capitalize(),
        test_mode=test_mode  # Pass the test_mode flag
    )

    # Run the experiment
    model_info = experiment.run()

    # Print or log the results as needed
    print(model_info)

if __name__ == "__main__":
    main()