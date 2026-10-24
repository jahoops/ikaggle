def check_dependencies():
    """Verify all required packages are installed"""
    packages = {
        'seaborn': 'sns',
        'matplotlib': 'plt',
        'pandas': 'pd',
        'numpy': 'np',
        'torch': 'torch',
        'cudf': 'cudf',
        'cuml': 'cuml',
        'xgboost': 'xgb',
        'lightgbm': 'lgb',
        'catboost': 'cb'
    }
    
    for package, alias in packages.items():
        try:
            exec(f"import {package} as {alias}")
            version = eval(f"{alias}.__version__")
            print(f"✓ {package:<12} {version}")
        except ImportError:
            print(f"✗ {package:<12} not found")

if __name__ == "__main__":
    check_dependencies()