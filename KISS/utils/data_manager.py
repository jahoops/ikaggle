import pandas as pd

def get_data(data_type='train', test_mode=False):
    """
    Load data based on type:
    - 'train': combined training data
    - 'just_train': only training set
    - 'validate': only validation set
    - 'contest': contest data
    """
    base_path = '/home/USER/Projects/ikaggle/KISS/data'
    
    if data_type == 'train':
        # Load and combine train and validate
        train_df = pd.read_csv(f'{base_path}/train.csv')
        validate_df = pd.read_csv(f'{base_path}/validate.csv')
        data = pd.concat([train_df, validate_df], ignore_index=True)
    elif data_type == 'just_train':
        data = pd.read_csv(f'{base_path}/train.csv')
    elif data_type == 'validate':
        data = pd.read_csv(f'{base_path}/validate.csv')
    elif data_type == 'contest':
        data = pd.read_csv(f'{base_path}/contest.csv')
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
    if test_mode:
        data = data.sample(frac=0.1, random_state=42)
    
    return data

# Define features and target
features = ['Age','AnnualIncome','NumberofDependents','HealthScore','PreviousClaims','VehicleAge','CreditScore','InsuranceDuration','SignupYear','SignupMonth','Gender_Female','MaritalStatus_Divorced','MaritalStatus_Married','MaritalStatus_Single','MaritalStatus_nan','EducationLevel_Bachelors','EducationLevel_HighSchool','EducationLevel_Masters','EducationLevel_PhD','Occupation_Employed','Occupation_SelfEmployed','Occupation_Unemployed','Occupation_nan','Location_Rural','Location_Suburban','Location_Urban','PolicyType_Basic','PolicyType_Comprehensive','PolicyType_Premium','CustomerFeedback_Average','CustomerFeedback_Good','CustomerFeedback_Poor','CustomerFeedback_nan','SmokingStatus_No','ExerciseFrequency_Daily','ExerciseFrequency_Monthly','ExerciseFrequency_Rarely','ExerciseFrequency_Weekly','PropertyType_Apartment','PropertyType_Condo','PropertyType_House']
target = 'PremiumAmount'
# Example usage:


