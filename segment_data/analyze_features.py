from pathlib import Path
import pandas as pd
from .feature_analyzer import FeatureAnalyzer
from .segment_optimizer import SegmentOptimizer
from experiments.data_manager import data_manager

def main():
    # Load data
    train_data = data_manager.finaltrain
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer(train_data)
    
    # Analyze each feature using optimal segments
    for feature in ['AnnualIncome', 'Age', 'HealthScore', 'CreditScore']:
        print(f"\nAnalyzing {feature} with optimal segments...")
        stats = analyzer.analyze_feature_impact(feature, use_optimal=True)
        
        # Create visualizations
        fig = analyzer.plot_feature_segments(feature, use_optimal=True)
        
        # Save results
        output_dir = Path('output/segment_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        stats.to_csv(output_dir / f"{feature}_optimal_analysis.csv")
        fig.savefig(output_dir / f"{feature}_optimal_segments.png")
        
        print(f"Optimal segments for {feature}:")
        optimal = analyzer.get_optimal_segments(feature)
        for label, (low, high) in zip(optimal['labels'], 
                                     zip(optimal['boundaries'][:-1], 
                                         optimal['boundaries'][1:])):
            print(f"{label}: {low:.2f} - {high:.2f}")

if __name__ == "__main__":
    main()