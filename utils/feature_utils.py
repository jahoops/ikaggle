# filepath: /home/USER/Kaggle/InsuranceNext/experiments/features.py

# Existing feature_importances list
feature_importances = [
    ("HealthScore", 0.5523),
    ("AnnualIncome", 0.2245),
    ("CreditScore", 0.0800),
    ("Age", 0.0085),
    ("SignupYear", 0.0023),
    ("PreviousClaims", 0.0021),
    ("NumberofDependents", 0.0018),
    ("Location_Rural", 0.0015),
    ("VehicleAge", 0.0015),
    ("InsuranceDuration", 0.0015),
    ("CustomerFeedback_Average", 0.0014),
    ("SmokingStatus_No", 0.0013),
    ("PropertyType_House", 0.0013),
    ("SignupMonth", 0.0012),
    ("Occupation_SelfEmployed", 0.0011),
    ("MaritalStatus_Married", 0.0011),
    ("Occupation_nan", 0.0009),
    ("CustomerFeedback_Good", 0.0007),
    ("EducationLevel_PhD", 0.0006),
    ("PropertyType_Apartment", 0.0005),
    ("Location_Urban", 0.0005),
    ("ExerciseFrequency_Daily", 0.0004)
]
# Core feature names (excluding distance or NN features)
feature_cols = [  'HealthScore'
                , 'CustomerFeedback_nan'
                , 'Age'
                , 'EducationLevel_Masters'
                , 'ExerciseFrequency_Rarely'
                , 'AnnualIncome'
                , 'MaritalStatus_Single'
                , 'MaritalStatus_nan'
                , 'CreditScore'
                , 'Occupation_SelfEmployed'
                , 'SignupMonth'
                , 'MaritalStatus_Divorced'
                , 'PolicyType_Comprehensive'
                , 'Occupation_Unemployed'
                , 'CustomerFeedback_Average'
                , 'MaritalStatus_Married'
                , 'EducationLevel_Bachelors'
                , 'ExerciseFrequency_Weekly'
                , 'Location_Rural'
                , 'PropertyType_Apartment'
                , 'NumberofDependents'
                , 'Location_Urban'
                , 'EducationLevel_HighSchool'
                , 'ExerciseFrequency_Monthly'
                , 'Occupation_nan'
                , 'Gender_Female'
                , 'SmokingStatus_No'
                , 'EducationLevel_PhD'
                , 'InsuranceDuration'
                , 'PolicyType_Basic'
                , 'ExerciseFrequency_Daily'
                , 'PreviousClaims'
                , 'CustomerFeedback_Good'
                , 'SignupYear'
                , 'VehicleAge'
                , 'PropertyType_House'
                , 'Occupation_Employed'
                ]

# Distance features
nn_dist = [f"DIST{i}" for i in range(1, 21)]

# Nearest neighbor features
nn_id = [f"NN{i}" for i in range(1, 21)]

# Combined features (core + distance + NN)
combined_features = feature_cols + nn_dist + nn_id

all_columns = ['Age', 'AnnualIncome', 'NumberofDependents', 'HealthScore',
       'PreviousClaims', 'VehicleAge', 'CreditScore', 'InsuranceDuration',
       'SignupYear', 'SignupMonth', 'MaritalStatus_Married',
       'EducationLevel_PhD', 'Occupation_SelfEmployed', 'Occupation_nan',
       'Location_Rural', 'Location_Urban', 'CustomerFeedback_Average',
       'CustomerFeedback_Good', 'SmokingStatus_No', 'ExerciseFrequency_Daily',
       'PropertyType_Apartment', 'PropertyType_House', 'PremiumAmount', 'id',
       'avg_dist','median_dist','avg_pred','median_pred']