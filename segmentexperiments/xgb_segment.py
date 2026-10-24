def get_experiment_metadata(self) -> Dict:
    """Return metadata for the experiment including GPU metrics and training scores"""
    metadata = {
        "name": "XGBoostSegmentExperiment",
        "test_mode": self.test_mode,
        "timestamp": datetime.now().isoformat(),
        "hardware_config": {
            "gpu_enabled": self.use_gpu,
            "device_name": None,
            "memory_allocated": None,
            "compute_capability": None
        }
    }
    
    # Add GPU metrics if available
    if self.use_gpu:
        try:
            import cupy as cp
            dev = cp.cuda.Device()
            
            # Get device properties using the correct CuPy API
            try:
                props = cp.cuda.runtime.getDeviceProperties(dev.id)
                device_name = props['name'].decode('utf-8')
                compute_major = props['major']
                compute_minor = props['minor']
                
                # Get memory information
                mem_info = dev.mem_info()
                free_memory, total_memory = mem_info
                
                metadata["hardware_config"].update({
                    "device_name": device_name,
                    "memory_allocated": float((total_memory - free_memory) / (1024**2)),
                    "memory_available": float(free_memory / (1024**2)),
                    "total_memory": float(total_memory / (1024**2)),
                    "compute_capability": f"{compute_major}.{compute_minor}"
                })
            except Exception as e:
                print(f"Warning: Could not get detailed GPU properties: {e}")
                # Fallback with basic information
                metadata["hardware_config"].update({
                    "device_name": f"CUDA Device {dev.id}",
                    "memory_allocated": float(dev.mem_info()[1] - dev.mem_info()[0]) / (1024**2),
                    "memory_available": float(dev.mem_info()[0]) / (1024**2),
                    "total_memory": float(dev.mem_info()[1]) / (1024**2),
                    "compute_capability": "0.0"
                })
            
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not get GPU metrics: {e}")
            
    # Add model information and training metrics
    if hasattr(self, 'model_trainer') and self.model_trainer is not None:
        metadata.update({
            "model_info": {
                "n_segments": len(self.model_trainer.models) if hasattr(self.model_trainer, 'models') else 0,
                "training_complete": hasattr(self.model_trainer, 'is_trained') and self.model_trainer.is_trained,
                "segment_scores": getattr(self.model_trainer, 'segment_scores', {})
            }
        })
    
    return metadata

def get_quick_test_params(self) -> Dict[str, Any]:
    """Return parameters for quick testing"""
    base_params = {
        'n_estimators': 200,  # Increased from default
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
        'random_state': 42
    }
    
    if self.test_mode:
        # Faster params for testing
        base_params.update({
            'n_estimators': 100,
            'max_depth': 4,
            'early_stopping_rounds': 10
        })
    else:
        # Full training params
        base_params.update({
            'n_estimators': 500,
            'max_depth': 8,
            'early_stopping_rounds': 50,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        })
    
    return base_params

def train(self, data: pd.DataFrame) -> None:
    """Train XGBoost model with validation"""
    super().train(data)
    
    # Split data for validation
    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]
    
    # Train with validation set
    self.model.fit(
        train_data[self.feature_order],
        train_data['PremiumAmount'],
        eval_set=[(val_data[self.feature_order], val_data['PremiumAmount'])],
        verbose=True
    )

def validate(self, val_data: pd.DataFrame) -> Dict[str, float]:
    """Validate model and return metrics with predictions"""
    if self.model is None:
        raise RuntimeError("Model must be trained before validation")
    
    # Keep original IDs
    ids = val_data['id'].copy() if 'id' in val_data.columns else None
    
    # Generate predictions
    X_val = val_data[self.feature_order]
    y_val = val_data['PremiumAmount']
    y_pred = self.model.predict(X_val)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_log_error
    metrics = {
        'val_rmsle': float(np.sqrt(mean_squared_log_error(y_val, y_pred))),
        'val_samples': len(y_val)
    }
    
    # Save predictions with IDs
    if ids is not None:
        predictions_df = pd.DataFrame({
            'id': ids,
            'PremiumAmount': y_pred
        })
        val_output_path = self.output_dir / f'val_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        predictions_df.to_csv(val_output_path, index=False)
        metrics['predictions_file'] = str(val_output_path)
    
    return metrics

def test(self, test_data: pd.DataFrame) -> Dict[str, float]:
    """Test model and return metrics with predictions"""
    if self.model is None:
        raise RuntimeError("Model must be trained before testing")
    
    # Keep original IDs
    ids = test_data['id'].copy() if 'id' in test_data.columns else None
    
    # Generate predictions
    X_test = test_data[self.feature_order]
    y_test = test_data['PremiumAmount']
    y_pred = self.model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_log_error, mean_squared_error
    metrics = {
        'test_rmsle': float(np.sqrt(mean_squared_log_error(y_test, y_pred))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'test_samples': len(y_test)
    }
    
    # Save predictions with IDs
    if ids is not None:
        predictions_df = pd.DataFrame({
            'id': ids,
            'PremiumAmount': y_pred
        })
        test_output_path = self.output_dir / f'test_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        predictions_df.to_csv(test_output_path, index=False)
        metrics['predictions_file'] = str(test_output_path)
    
    return metrics