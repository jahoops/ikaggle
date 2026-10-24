OPTIMAL_SEGMENTS = {
    'AnnualIncome': {
        'method': 'quantile',
        'n_segments': 9,
        'boundaries': [
            1.0, 2918.0, 6973.0, 14430.0, 20360.0, 
            26139.0, 35407.0, 45907.0, 74099.0, 149997.0
        ],
        'labels': [
            'Very Low',     # 1.0 - 2918.0
            'Low',          # 2918.0 - 6973.0
            'Lower Middle', # 6973.0 - 14430.0
            'Middle',       # 14430.0 - 20360.0
            'Upper Middle', # 20360.0 - 26139.0
            'Lower High',   # 26139.0 - 35407.0
            'High',         # 35407.0 - 45907.0
            'Very High',    # 45907.0 - 74099.0
            'Maximum'       # 74099.0 - 149997.0
        ]  # 9 labels for 10 boundaries ✓
    },
    'Age': {
        'method': 'quantile',
        'n_segments': 10,
        'boundaries': [
            18.0, 22.0, 27.0, 32.0, 37.0, 41.0, 
            46.0, 50.0, 55.0, 60.0, 64.0
        ],
        'labels': [
            'Young Adult',  # 18.0 - 22.0
            'Early 20s',    # 22.0 - 27.0
            'Late 20s',     # 27.0 - 32.0
            'Early 30s',    # 32.0 - 37.0
            'Late 30s',     # 37.0 - 41.0
            'Early 40s',    # 41.0 - 46.0
            'Mid 40s',      # 46.0 - 50.0
            'Early 50s',    # 50.0 - 55.0
            'Mid 50s',      # 55.0 - 60.0
            'Senior'        # 60.0 - 64.0
        ]  # 10 labels for 11 boundaries ✓
    },
    'HealthScore': {
        'method': 'quantile',
        'n_segments': 9,
        'boundaries': [
            2.01, 10.84, 15.47, 19.69, 23.64, 
            25.53, 29.92, 35.08, 41.96, 58.98
        ],
        'labels': [
            'Very Poor',    # 2.01 - 10.84
            'Poor',         # 10.84 - 15.47
            'Below Avg',    # 15.47 - 19.69
            'Slightly Low', # 19.69 - 23.64
            'Average',      # 23.64 - 25.53
            'Above Avg',    # 25.53 - 29.92
            'Good',         # 29.92 - 35.08
            'Very Good',    # 35.08 - 41.96
            'Excellent'     # 41.96 - 58.98
        ]  # 9 labels for 10 boundaries ✓
    },
    'CreditScore': {
        'method': 'kmeans',
        'n_segments': 8,
        'boundaries': [
            300.0, 343.69, 429.29, 503.35, 586.61,
            641.37, 700.29, 761.57, 819.98, 849.0
        ],
        'labels': [
            'Very Poor',    # 300.0 - 343.69
            'Poor',         # 343.69 - 429.29
            'Fair',         # 429.29 - 503.35
            'Below Avg',    # 503.35 - 586.61
            'Average',      # 586.61 - 641.37
            'Good',         # 641.37 - 700.29
            'Very Good',    # 700.29 - 761.57
            'Excellent',    # 761.57 - 819.98
            'Maximum'       # 819.98 - 849.0
        ]  # 9 labels for 10 boundaries ✓
    }
}