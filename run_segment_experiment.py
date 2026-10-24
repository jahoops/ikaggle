from pathlib import Path
from segmentexperiments.xgboost_segment import XGBoostSegmentExperiment
from experiments.data_manager import data_manager

def main():
    experiment = XGBoostSegmentExperiment(
        output_dir=Path('output/segments'),
        test_mode=False
    )
    
    experiment.run(
        train_data=data_manager.finaltrain,
        val_data=data_manager.finalvalidate,
        test_data=data_manager.finaltest
    )

if __name__ == "__main__":
    main()