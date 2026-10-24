import numpy as np
from numba import jit  # For CPU acceleration

FEATURE_CONFIGS = {
    'Age': {
        'type': 'binned',
        'bins': [
            (18, 25, 0.2),   # Young adults - lower weight
            (26, 35, 0.4),   # Early career - medium weight
            (36, 50, 0.6),   # Mid career - higher weight
            (51, 65, 0.8),   # Late career - high weight
            (66, 85, 1.0)    # Retirement - highest weight
        ],
        'norm_offset': 18,
        'norm_scale': 67,
        'interpolate': True  # Enable interpolation between bins
    },
    'AnnualIncome': {
        'type': 'binned',
        'bins': [
            (0, 30000, 0.2),      # Low income
            (30001, 60000, 0.4),  # Lower middle
            (60001, 100000, 0.6), # Middle
            (100001, 150000, 0.8),# Upper middle
            (150001, 200000, 1.0) # High income
        ],
        'norm_scale': 200000,
        'interpolate': True
    },
    'HealthScore': {
        'type': 'binned',
        'bins': [
            (0, 20, 0.2),   # Poor health
            (21, 40, 0.4),  # Fair health
            (41, 60, 0.6),  # Average health
            (61, 80, 0.8),  # Good health
            (81, 100, 1.0)  # Excellent health
        ],
        'norm_scale': 100,
        'interpolate': True
    },
    'NumberofDependents': {
        'type': 'variable',
        'min': 0,
        'max': 1,
        'step': 0.1,
        # For raw data: offset=0, scale=10
        'norm_offset': 0,
        'norm_scale': 10,
        'translate': lambda x: int(round(x * 10))
    },
    'PreviousClaims': {
        'type': 'variable',
        'min': 0,
        'max': 1,
        'step': 0.1,
        # For raw data: offset=0, scale=10
        'norm_offset': 0,
        'norm_scale': 10,
        'translate': lambda x: int(round(x * 10))
    },
    'VehicleAge': {
        'type': 'variable',
        'min': 0,
        'max': 1,
        'step': 0.04,
        # For raw data: offset=0, scale=25
        'norm_offset': 0,
        'norm_scale': 25,
        'translate': lambda x: x * 25
    },
    'CreditScore': {
        'type': 'variable',
        'min': 0,
        'max': 1,
        'step': 0.001,
        # For raw data: offset=300, scale=550 (since 850-300=550)
        'norm_offset': 300,
        'norm_scale': 550,
        'translate': lambda x: x * 550 + 300
    },
    'InsuranceDuration': {
        'type': 'variable',
        'min': 0,
        'max': 1,
        'step': 0.05,
        # For raw data: offset=0, scale=20
        'norm_offset': 0,
        'norm_scale': 20,
        'translate': lambda x: x * 20
    },
    'SignupYear': {
        'type': 'variable',
        'min': 0,
        'max': 1,
        'step': 0.2,
        # For raw data: offset=2019, scale=5  (from 2019 to 2024)
        'norm_offset': 2019,
        'norm_scale': 5,
        'translate': lambda x: x * 5 + 2019
    },
    'SignupMonth': {
        'type': 'variable',
        'min': 0,
        'max': 1,
        'step': 0.0834,  # roughly 1/12
        # For raw data: offset=1, scale=11  (1 to 12)
        'norm_offset': 1,
        'norm_scale': 11,
        'translate': lambda x: int(round(x * 11 + 1))
    },
    
    # Convert categorical features to ordinal scales
    'EducationLevel': {
        'type': 'ordinal',
        'mapping': {
            'nan': 0.0,
            'HighSchool': 0.25,
            'Bachelors': 0.5,
            'Masters': 0.75,
            'PhD': 1.0
        }
    },
    'Occupation': {
        'type': 'ordinal',
        'mapping': {
            'nan': 0.0,
            'Unemployed': 0.33,
            'Employed': 0.67,
            'SelfEmployed': 1.0
        }
    },
    'Location': {
        'type': 'ordinal',
        'mapping': {
            'Rural': 0.33,
            'Suburban': 0.67,
            'Urban': 1.0
        }
    },
    'PolicyType': {
        'type': 'ordinal',
        'mapping': {
            'Basic': 0.33,
            'Comprehensive': 0.67,
            'Premium': 1.0
        }
    },
    'CustomerFeedback': {
        'type': 'ordinal',
        'mapping': {
            'nan': 0.0,
            'Poor': 0.33,
            'Average': 0.67,
            'Good': 1.0
        }
    },
    'ExerciseFrequency': {
        'type': 'ordinal',
        'mapping': {
            'Rarely': 0.25,
            'Monthly': 0.5,
            'Weekly': 0.75,
            'Daily': 1.0
        }
    },
    'PropertyType': {
        'type': 'ordinal',
        'mapping': {
            'Apartment': 0.33,
            'Condo': 0.67,
            'House': 1.0
        }
    },
    'MaritalStatus': {
        'type': 'ordinal',
        'mapping': {
            'nan': 0.0,
            'Single': 0.33,
            'Divorced': 0.67,
            'Married': 1.0
        }
    },
    
    # Keep some features as true/false
    'Gender': {
        'type': 'boolean',
        'mapping': {
            'Female': 1,
            'Male': 0
        }
    },
    'SmokingStatus': {
        'type': 'boolean',
        'mapping': {
            'No': 0,
            'Yes': 1
        }
    }
}

def get_gene_length(feature_name):
    """Calculate number of bits needed for a feature's gene using binned ranges"""
    config = FEATURE_CONFIGS[feature_name]
    if config['type'] == 'boolean':
        return 1
    elif config['type'] == 'binned':
        # Number of bits needed to represent n bins is log2(n)
        n_bins = len(config['bins'])
        return max(1, int(np.ceil(np.log2(n_bins))))
    else:
        # Fallback for continuous variables
        if 'norm_scale' in config:
            real_range = config['norm_scale']
            # Use coarser steps for continuous variables
            real_step = config['step'] * real_range * 5  # increased step size
            steps = real_range / real_step
        else:
            value_range = config['max'] - config['min']
            steps = value_range / config['step']
        return max(1, int(np.ceil(np.log2(steps + 1))))

def decode_gene(gene_bits, feature_name):
    """Convert binary gene representation to actual feature value"""
    config = FEATURE_CONFIGS[feature_name]
    if config['type'] == 'boolean':
        return bool(gene_bits[0])
    
    # Convert binary array to integer.
    value = int(''.join(map(str, gene_bits)), 2)
    
    # Scale to normalized range (0-1)
    max_gene_value = (1 << len(gene_bits)) - 1
    norm_value = value / max_gene_value
    
    # Use translate function if available
    if 'translate' in config:
        return config['translate'](norm_value)
    else:
        return norm_value * (config['max'] - config['min']) + config['min']

def decode_ordinal_feature(value, feature_name):
    """Convert categorical value to ordinal scale"""
    config = FEATURE_CONFIGS[feature_name]
    if config['type'] != 'ordinal':
        return value
    return config['mapping'].get(value, 0.0)

@jit(nopython=True)
def _compute_derived_feature_numba(feature_type, base_value1, base_value2=None):
    """Accelerated computation of derived features"""
    if feature_type == 0:
        return base_value1 * base_value1
    elif feature_type == 1:
        return base_value1 * base_value2
    elif feature_type == 2:
        return base_value1 / max(1, base_value2)
    return 0

def compute_derived_feature(feature_name, feature_values):
    """Compute derived features using acceleration where applicable."""
    if feature_name.endswith('_Squared'):
        base_feature = feature_name.replace('_Squared', '')
        return _compute_derived_feature_numba(0, feature_values[base_feature])
    elif '_Interaction' in feature_name:
        f1, f2 = feature_name.split('_Interaction')[0].split('_')
        return _compute_derived_feature_numba(1, feature_values[f1], feature_values[f2])
    elif feature_name in ['IncomePerDependent', 'HealthAgeIndex', 'CreditIncomeRatio']:
        if feature_name == 'IncomePerDependent':
            return _compute_derived_feature_numba(2, feature_values['AnnualIncome'], feature_values['NumberofDependents'])
        elif feature_name == 'HealthAgeIndex':
            return _compute_derived_feature_numba(2, feature_values['HealthScore'], feature_values['Age'])
        else:
            return _compute_derived_feature_numba(2, feature_values['CreditScore'], feature_values['AnnualIncome'])
    return 0

def get_bin_value(value, config):
    """Get interpolated value for binned features"""
    if not config['interpolate']:
        # Traditional binning
        for low, high, weight in config['bins']:
            if low <= value <= high:
                return weight
        return 0.0
    
    # Interpolated binning
    for i, (low, high, weight) in enumerate(config['bins']):
        if low <= value <= high:
            if i == 0:  # First bin
                next_low, _, next_weight = config['bins'][i + 1]
                factor = (value - low) / (next_low - low)
                return weight + factor * (next_weight - weight)
            elif i == len(config['bins']) - 1:  # Last bin
                prev_high, _, prev_weight = config['bins'][i - 1]
                factor = (value - prev_high) / (high - prev_high)
                return prev_weight + factor * (weight - prev_weight)
            else:  # Middle bins
                prev_high, _, prev_weight = config['bins'][i - 1]
                next_low, _, next_weight = config['bins'][i + 1]
                if value <= (low + high) / 2:
                    factor = (value - prev_high) / (low - prev_high)
                    return prev_weight + factor * (weight - prev_weight)
                else:
                    factor = (value - high) / (next_low - high)
                    return weight + factor * (next_weight - weight)
    return 0.0

# Update feature engineering functions
def encode_binned_feature(value, feature_name):
    """Convert binned feature to continuous value"""
    config = FEATURE_CONFIGS[feature_name]
    if config['type'] != 'binned':
        return value
        
    # Normalize raw value
    if 'norm_offset' in config:
        value = value - config['norm_offset']
    if 'norm_scale' in config:
        value = value / config['norm_scale']
        
    return get_bin_value(value, config)
