import joblib
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def display_data(obj, max_lines=100):
    """Print the object contents, limiting repetitive lines to max_lines."""
    if isinstance(obj, pd.DataFrame):
        print("DataFrame shape:", obj.shape)
        print(obj.head(max_lines))
    elif isinstance(obj, pd.Series):
        print("Series shape:", obj.shape)
        print(obj.head(max_lines))
    elif isinstance(obj, dict):
        print("Dictionary keys:", list(obj.keys()))
        for k, v in obj.items():
            print(f"Key: {k}")
            if isinstance(v, (list, pd.DataFrame, pd.Series)):
                print(f"Value type: {type(v)} (showing up to {max_lines} lines)")
                if isinstance(v, pd.DataFrame):
                    print("DataFrame shape:", v.shape)
                    print(v.head(max_lines))
                elif isinstance(v, pd.Series):
                    print("Series shape:", v.shape)
                    print(v.head(max_lines))
                else:
                    print(v[:max_lines])
            else:
                print(f"Value: {v}")
    elif isinstance(obj, list):
        print(f"List of length {len(obj)} (showing up to {max_lines} items):")
        print(obj[:max_lines])
    else:
        print("Object type:", type(obj))
        print(obj)

def main():
    if len(sys.argv) < 2:
        print("Usage: python pkl_inspector.py <path_to_pkl_file>")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    data = joblib.load(pkl_path)
    display_data(data, max_lines=100)

    # Example: remove or rename "transform_input" in your pipeline
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        # No transform_input here
        ('model', SomeRegressor())
    ])

if __name__ == "__main__":
    main()