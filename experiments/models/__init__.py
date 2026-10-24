import logging
import os

logger = logging.getLogger(__name__)

def get_tf_registry():
    """Get TensorFlow-dependent models"""
    try:
        # Only attempt to import TF-dependent code if present
        from .neuralnet import NeuralNetExperiment
        from xgboost import XGBRegressor  # For XGBoostExperiment
        from .xgboost import XGBoostExperiment
        return {
            'neuralnet': NeuralNetExperiment,
            'xgboost': XGBoostExperiment
        }
    except ImportError as e:
        logger.warning(f"TF models not available: {e}")
        return {}

def get_base_registry():
    """Get non-TF models"""
    from .catboost import CatBoostExperiment
    from .knn import KNNExperiment
    from .lasso import LassoExperiment
    from .lightgbm import LightGBMExperiment
    from .linear import LinearRegressionExperiment, RidgeRegressionExperiment
    from .randomforest import RandomForestExperiment
    # from .svr import SVRExperiment
    
    return {
        'catboost': CatBoostExperiment,
        'knn': KNNExperiment,
        'lasso': LassoExperiment,
        'lightgbm': LightGBMExperiment,
        'linear': LinearRegressionExperiment,
        'ridge': RidgeRegressionExperiment,
        'randomforest': RandomForestExperiment,
        # 'svr': SVRExperiment
    }

# Build registry based on environment
MODEL_REGISTRY = get_base_registry()
# current_venv = os.environ.get('VIRTUAL_ENV', '').split('/')[-1]

# if current_venv == 'vnext':
    # Merge TF registry only if in 'vnext' environment

MODEL_REGISTRY.update(get_tf_registry())

# Safe imports for all experiments—only used if registry demands them
try:
    from .catboost import CatBoostExperiment
    from .knn import KNNExperiment
    from .lasso import LassoExperiment
    from .lightgbm import LightGBMExperiment
    from .linear import LinearRegressionExperiment, RidgeRegressionExperiment
    from .randomforest import RandomForestExperiment
    from .neuralnet import NeuralNetExperiment
    from .svr import SVRExperiment
    from ._xgboost import XGBoostExperiment
except ImportError as e:
    logger.debug(f"Optional module not available: {e}")

__all__ = list(MODEL_REGISTRY.keys())
