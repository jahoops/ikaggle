import sqlite3
import csv
import os

conn = sqlite3.connect("kaggle.db")
cur = conn.cursor()

# Drop table if it exists
cur.execute("DROP TABLE IF EXISTS insurance")

# Create table with specified columns
cur.execute('''
    CREATE TABLE insurance (
        Age FLOAT,
        AnnualIncome FLOAT,
        NumberofDependents FLOAT,
        HealthScore FLOAT,
        PreviousClaims FLOAT,
        VehicleAge FLOAT,
        CreditScore FLOAT,
        InsuranceDuration FLOAT,
        SignupYear FLOAT,
        SignupMonth FLOAT,
        Gender_Female FLOAT,
        MaritalStatus_Divorced FLOAT,
        MaritalStatus_Married FLOAT,
        MaritalStatus_Single FLOAT,
        MaritalStatus_nan FLOAT,
        EducationLevel_Bachelors FLOAT,
        EducationLevel_HighSchool FLOAT,
        EducationLevel_Masters FLOAT,
        EducationLevel_PhD FLOAT,
        Occupation_Employed FLOAT,
        Occupation_SelfEmployed FLOAT,
        Occupation_Unemployed FLOAT,
        Occupation_nan FLOAT,
        Location_Rural FLOAT,
        Location_Suburban FLOAT,
        Location_Urban FLOAT,
        PolicyType_Basic FLOAT,
        PolicyType_Comprehensive FLOAT,
        PolicyType_Premium FLOAT,
        CustomerFeedback_Average FLOAT,
        CustomerFeedback_Good FLOAT,
        CustomerFeedback_Poor FLOAT,
        CustomerFeedback_nan FLOAT,
        SmokingStatus_No FLOAT,
        ExerciseFrequency_Daily FLOAT,
        ExerciseFrequency_Monthly FLOAT,
        ExerciseFrequency_Rarely FLOAT,
        ExerciseFrequency_Weekly FLOAT,
        PropertyType_Apartment FLOAT,
        PropertyType_Condo FLOAT,
        PropertyType_House FLOAT,
        PremiumAmount FLOAT,
        RecordType INTEGER,
        id INTEGER PRIMARY KEY
    )
''')

def import_cleaned_data():
    """
    Imports rows from three cleaned CSV files into the insurance table.
    Assumes:
      • cleaned_train.csv has PremiumAmount in the second-to-last column, and ID in the last column (RecordType is not in file).
      • cleaned_validate.csv is similar, but we set RecordType=1 for these rows.
      • cleaned_test.csv does NOT have PremiumAmount, so we set PremiumAmount=0, and RecordType=2.
    
    Final layout in the insurance table (42 columns + RecordType + ID = 44 total):
      1)  Age
      2)  AnnualIncome
      3)  NumberofDependents
      4)  HealthScore
      5)  PreviousClaims
      6)  VehicleAge
      7)  CreditScore
      8)  InsuranceDuration
      9)  SignupYear
      10) SignupMonth
      11) Gender_Female
      12) MaritalStatus_Divorced
      13) MaritalStatus_Married
      14) MaritalStatus_Single
      15) MaritalStatus_nan
      16) EducationLevel_Bachelors
      17) EducationLevel_HighSchool
      18) EducationLevel_Masters
      19) EducationLevel_PhD
      20) Occupation_Employed
      21) Occupation_SelfEmployed
      22) Occupation_Unemployed
      23) Occupation_nan
      24) Location_Rural
      25) Location_Suburban
      26) Location_Urban
      27) PolicyType_Basic
      28) PolicyType_Comprehensive
      29) PolicyType_Premium
      30) CustomerFeedback_Average
      31) CustomerFeedback_Good
      32) CustomerFeedback_Poor
      33) CustomerFeedback_nan
      34) SmokingStatus_No
      35) ExerciseFrequency_Daily
      36) ExerciseFrequency_Monthly
      37) ExerciseFrequency_Rarely
      38) ExerciseFrequency_Weekly
      39) PropertyType_Apartment
      40) PropertyType_Condo
      41) PropertyType_House
      42) PremiumAmount
      43) RecordType
      44) id
    """
    file_info = [
        ("data/processed/cleaned_train.csv", 0),
        ("data/processed/cleaned_validate.csv", 1),
        ("data/processed/cleaned_test.csv", 2)
    ]

    insert_query = '''
        INSERT INTO insurance (
            Age,AnnualIncome,NumberofDependents,HealthScore,PreviousClaims,VehicleAge,
            CreditScore,InsuranceDuration,SignupYear,SignupMonth,Gender_Female,
            MaritalStatus_Divorced,MaritalStatus_Married,MaritalStatus_Single,
            MaritalStatus_nan,EducationLevel_Bachelors,EducationLevel_HighSchool,
            EducationLevel_Masters,EducationLevel_PhD,Occupation_Employed,
            Occupation_SelfEmployed,Occupation_Unemployed,Occupation_nan,
            Location_Rural,Location_Suburban,Location_Urban,
            PolicyType_Basic,PolicyType_Comprehensive,PolicyType_Premium,
            CustomerFeedback_Average,CustomerFeedback_Good,CustomerFeedback_Poor,
            CustomerFeedback_nan,SmokingStatus_No,ExerciseFrequency_Daily,
            ExerciseFrequency_Monthly,ExerciseFrequency_Rarely,ExerciseFrequency_Weekly,
            PropertyType_Apartment,PropertyType_Condo,PropertyType_House,
            PremiumAmount,RecordType,id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    '''

    for file_path, record_type in file_info:
        if not os.path.exists(file_path):
            print(f"{file_path} not found. Skipping.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # skip header

            rows = []
            for row in reader:
                # For test data (record_type=2), no PremiumAmount column
                if record_type == 2:
                    # Take exactly the first 41 numeric columns
                    numeric_values = [float(x) for x in row[:41]]
                    # Append PremiumAmount=0
                    numeric_values.append(0.0)
                    # Append RecordType=2
                    numeric_values.append(2)
                    # Append id from the last column
                    numeric_values.append(int(row[-1]))
                else:
                    # For train/validate: last two columns = PremiumAmount, id
                    numeric_values = [float(x) for x in row[:-2]]
                    numeric_values.append(float(row[-2]))  # PremiumAmount
                    numeric_values.append(record_type)
                    numeric_values.append(int(row[-1]))    # id

                rows.append(numeric_values)

        cur.executemany(insert_query, rows)
        conn.commit()
        print(f"Imported {len(rows)} rows from {file_path}")

import_cleaned_data()

# Commit and close
conn.commit()
conn.close()
