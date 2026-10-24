import numpy as np

class FeatureCombiner:
    """Handles feature combinations with richer operations"""
    
    @staticmethod
    def get_operation_code(bits):
        """Convert 3 bits to operation name"""
        code = int(''.join(map(str, bits)), 2)
        operations = ['add', 'subtract', 'multiply', 'divide', 'max', 'min', 'ratio', 'diff']
        return operations[code % len(operations)]
    
    @staticmethod
    def combine_features(f1, f2, operation):
        """Combine two features using specified operation with safety checks"""
        try:
            # Convert and clean inputs
            f1 = np.asarray(f1, dtype=np.float32)
            f2 = np.asarray(f2, dtype=np.float32)
            
            # Replace inf/nan values
            f1 = np.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)
            f2 = np.nan_to_num(f2, nan=0.0, posinf=0.0, neginf=0.0)
            
            eps = 1e-6  # Small value to prevent division by zero
            
            if operation == 'add':
                result = (f1 + f2) / 2  # Normalize to keep in [0,1]
            elif operation == 'subtract':
                result = (f1 - f2 + 1) / 2  # Shift and scale to [0,1]
            elif operation == 'multiply':
                result = f1 * f2  # Already in [0,1] if inputs are
            elif operation == 'divide':
                # Safe division, result normalized to [0,1]
                result = f1 / (f2 + eps)
                result = (result - result.min()) / (result.max() - result.min() + eps)
            elif operation == 'max':
                result = np.maximum(f1, f2)
            elif operation == 'min':
                result = np.minimum(f1, f2)
            elif operation == 'ratio':
                # Normalized ratio
                ratio = f1 / (f2 + eps)
                result = (ratio - ratio.min()) / (ratio.max() - ratio.min() + eps)
            elif operation == 'diff':
                # Absolute difference, normalized
                result = np.abs(f1 - f2)
            else:
                result = (f1 + f2) / 2  # Default to average
            
            # Ensure result is finite and in [0,1]
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            return np.clip(result, 0, 1)
            
        except Exception as e:
            print(f"Feature combination error: {str(e)}")
            print(f"Operation: {operation}")
            print(f"Shapes: f1={f1.shape}, f2={f2.shape}")
            return np.zeros_like(f1)
